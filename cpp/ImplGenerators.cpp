// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ImplGenerators.hpp"
#include "TupleEnums.hpp"
#include "TupleOps.hpp"
#include "TupleTypes.hpp"
#include <ImplResolution.hpp>

namespace mlir::tuple {

template<class Range> SmallVector<Type> toTypes(const Range& types) {
  return llvm::map_to_vector(types, [](auto ty) {
    return Type(ty);
  });
}

/// The pair of tuple types a generator answers a two-argument demand with, or
/// failure when the demand is not two tuples of equal arity, or still mentions
/// a type variable.
///
/// A generator answers for the application it was asked about and no other, so
/// a demand that instantiation has not yet made concrete gets no impl: the
/// obligation stays pending until the types it maps over are known.
static FailureOr<std::pair<TupleType, TupleType>>
groundTuplePair(trait::ClaimType wanted) {
  auto typeArgs = wanted.getTraitApplication().getTypeArgs();
  if (typeArgs.size() != 2)
    return failure();

  if (llvm::any_of(typeArgs, [](Type ty) { return trait::isPolymorphicType(ty); }))
    return failure();

  auto lhs = dyn_cast<TupleType>(typeArgs[0]);
  auto rhs = dyn_cast<TupleType>(typeArgs[1]);
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return failure();

  return std::make_pair(lhs, rhs);
}

static LogicalResult matchGenerator(trait::TraitOp trait,
                                    trait::ClaimType wanted,
                                    StringRef whichGenerator,
                                    unsigned int minNumTypeArgs,
                                    unsigned int maxNumTypeArgs = UINT_MAX) {
  // check that the trait needs the impl generator of interest
  auto tag = trait->getAttrOfType<StringAttr>("tuple.impl_generator");
  if (!tag || tag.getValue() != whichGenerator)
    return failure();

  // confirm the wanted application targets this trait
  auto app = wanted.getTraitApplication();
  if (app.getTraitName().getValue() != trait.getSymName())
    return failure();

  unsigned int numTypeArgs = app.getTypeArgs().size();
  if (numTypeArgs < minNumTypeArgs || numTypeArgs > maxNumTypeArgs)
    return failure();

  return success();
}

static FailureOr<Type> homogeneousTupleElement(Type ty) {
  auto tupleTy = dyn_cast<TupleType>(ty);
  if (!tupleTy)
    return failure();

  // The empty tuple is homogeneous with uninhabited element type `none`.
  if (tupleTy.size() == 0)
    return NoneType::get(ty.getContext());

  // Non-empty tuples are homogeneous only when every element type matches
  // the first element exactly.
  Type element = tupleTy.getType(0);
  for (Type current : tupleTy.getTypes().drop_front())
    if (current != element)
      return failure();

  return element;
}

static SmallVector<trait::ClaimType> mapTraitAcrossTupleElements(
    FlatSymbolRefAttr traitRef,
    TupleType lhs,
    TupleType rhs) {
  using namespace mlir::trait;

  // given lhs := tuple<L0..Lm> and rhs := tuple<R0..Rm>,
  // we want to produce a list of ClaimTypes:
  //
  // !trait.claim<@Trait[L0,R0]> .. !trait.claim<@Trait[Lm,Rm]>

  MLIRContext *ctx = lhs.getContext();
  return llvm::map_to_vector(
    llvm::zip_equal(lhs.getTypes(), rhs.getTypes()),
    [&](auto elements) {
      auto [Li, Ri] = elements;
      return ClaimType::get(ctx, traitRef, {Li, Ri});
    });
}

/// TupleGenerator synthesizes impls for traits that declare themselves as the
/// structural "this type is a tuple" predicate.
///
/// The generated trait must have exactly one type argument. The impl is emitted
/// only when the wanted self type is a concrete TupleType; opaque types need an
/// explicit source-level bound instead of guessing tuple structure here.
struct TupleGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               OpBuilder &builder) const override {
    using namespace mlir::trait;

    // The source bridge selects this generator with tuple.impl_generator =
    // "tuple"; unrelated traits are left to other generators.
    if (failed(matchGenerator(trait, wanted, "tuple", 1, 1)))
      return failure();

    // Only concrete tuple types are structurally known to satisfy this trait.
    // Polymorphic or opaque types need an explicit bound from the source side.
    Type selfTy = wanted.getTraitApplication().getTypeArgs().front();
    if (!isa<TupleType>(selfTy))
      return failure();

    // Generated impls live at module scope and use the same canonical symbol
    // name as parser-created trait.impl operations.
    ModuleOp module = trait->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    MLIRContext *ctx = builder.getContext();
    Location loc = builder.getUnknownLoc();
    auto traitRef = FlatSymbolRefAttr::get(ctx, trait.getSymName());
    auto claim = ClaimType::get(ctx, traitRef, {selfTy});

    // Tuple has no methods or associated types; the body alleges whatever the
    // trait requires, for selection to prove where a use reads it.
    ImplOp impl = ImplOp::create(builder, loc, claim.getTraitApplication(),
                                 ArrayRef<ClaimType>{});
    allegeRequirements(impl, trait, builder);
    return impl;
  }
};

/// HomogeneousTupleGenerator synthesizes impls for homogeneous tuple facts.
///
/// The generated trait has one type argument and an `Element` associated type.
/// A concrete tuple matches when every element has the same type. The
/// empty tuple uses `none` as its uninhabited element type.
struct HomogeneousTupleGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               OpBuilder &builder) const override {
    using namespace mlir::trait;

    // The source bridge selects this generator with tuple.impl_generator =
    // "homogeneous_tuple"; unrelated traits are left to other generators.
    if (failed(matchGenerator(trait, wanted, "homogeneous_tuple", 1, 1)))
      return failure();

    // Match only concrete homogeneous tuple types and compute the Element
    // associated type at the same time.
    auto typeArgs = wanted.getTraitApplication().getTypeArgs();
    Type tupleTy = typeArgs[0];
    auto elementTy = homogeneousTupleElement(tupleTy);
    if (failed(elementTy))
      return failure();

    // Generated impls live at module scope and use the same canonical symbol
    // name as parser-created trait.impl operations.
    ModuleOp module = trait->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    MLIRContext *ctx = builder.getContext();
    Location loc = builder.getUnknownLoc();
    auto traitRef = FlatSymbolRefAttr::get(ctx, trait.getSymName());
    auto claim = ClaimType::get(ctx, traitRef, {tupleTy});

    // The impl is built whole on a detached builder and inserted once, so a
    // half-built impl is never a candidate. Its body binds Element and alleges
    // what the trait requires, for selection to prove where a use reads it.
    OpBuilder detached(ctx);
    ImplOp impl = ImplOp::create(detached, loc, claim.getTraitApplication(),
                                 ArrayRef<ClaimType>{});

    detached.setInsertionPointToStart(&impl.getBody().front());
    AssocTypeOp::create(detached, loc, "Element",
                        TypeAttr::get(*elementTy), ArrayAttr{});
    allegeRequirements(impl, trait, detached);

    builder.insert(impl);
    return impl;
  }
};

/// MapGenerator synthesizes the `trait.impl` of a "map" trait: the trait that
/// maps another trait across the elements of two tuples and binds the tuple of
/// the resulting claims as its `Claims` associated type.
///
///   trait.trait @tuple.MapEq(%self: !trait.claim<@tuple.MapEq[!S, !O]>) attributes {
///     tuple.impl_generator = "map",
///     tuple.mapped_trait   = @Eq
///   } {
///     trait.assoc_type @Claims
///     trait.method @claims() -> !trait.proj<@tuple.MapEq[!S,!O], "Claims">
///   }
///
/// For a wanted claim like:
///
///   !trait.claim<@tuple.MapEq[tuple<i32>, tuple<i64>]>
///
/// this generator answers with the impl for exactly that application:
///
///   trait.impl @...(%self: !trait.claim<@tuple.MapEq[tuple<i32>, tuple<i64>]>,
///                   %eq: !trait.claim<@Eq[i32, i64]>) {
///       trait.assoc_type @Claims = tuple<!trait.claim<@Eq[i32,i64]>>
///       trait.method @claims() -> tuple<!trait.claim<@Eq[i32,i64]>> {
///         %res = tuple.make(%eq)
///         trait.return %res
///       }
///     }
///
/// Position i of the two tuples contributes the where entry @Eq[Li, Ri], and
/// the `Claims` binding is those same applications claimed, in order. A demand
/// that is not two tuples of equal arity, or that still mentions a type
/// variable, gets no impl.
struct MapGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               OpBuilder &builder) const override {
    using namespace mlir::trait;

    // The source bridge selects this generator with tuple.impl_generator =
    // "map"; the mapper is applied to the two tuple types it maps over.
    if (failed(matchGenerator(trait, wanted, "map", 2, 2)))
      return failure();

    // the tuple.mapped_trait attribute must exist
    auto mappedTrait = trait->getAttrOfType<FlatSymbolRefAttr>("tuple.mapped_trait");
    if (!mappedTrait) return failure();

    // the demanded tuple types are what the trait is mapped across
    auto tuples = groundTuplePair(wanted);
    if (failed(tuples)) return failure();
    auto [lhs, rhs] = *tuples;

    // Generated impls live at module scope and use the same canonical symbol
    // name as parser-created trait.impl operations.
    ModuleOp module = trait->getParentOfType<ModuleOp>();
    if (!module) return failure();

    MLIRContext *ctx = builder.getContext();
    Location loc = builder.getUnknownLoc();

    // the mapped trait applied to each element position in turn:
    // !trait.claim<@MappedTrait[L0,R0]> .. !trait.claim<@MappedTrait[Lm,Rm]>
    SmallVector<ClaimType> claims = mapTraitAcrossTupleElements(mappedTrait, lhs, rhs);
    TupleType tupleOfClaims = TupleType::get(ctx, toTypes(claims));

    // the impl answers the demanded application itself
    auto selfApp = TraitApplicationAttr::get(
      ctx,
      FlatSymbolRefAttr::get(ctx, trait.getSymName()),
      ArrayRef<Type>{lhs, rhs}
    );

    // The impl is built whole on a detached builder and inserted once, so a
    // half-built impl is never a candidate. Its where entries are those same
    // applications, one per position.
    OpBuilder detached(ctx);
    ImplOp impl = ImplOp::create(detached, loc, selfApp, claims);

    // the impl's body binds @Claims and defines @claims(), whose result is the
    // tuple of the impl's where entries, its block arguments after the first:
    //   %res = tuple.make(%c0, %c1, ...)
    //   trait.return %res
    {
      detached.setInsertionPointToStart(&impl.getBody().front());

      AssocTypeOp::create(detached, loc, "Claims",
                          TypeAttr::get(tupleOfClaims), ArrayAttr{});

      // trait.method @claims() -> tupleOfClaims
      auto claimsFunc = trait::MethodOp::create(detached,
        loc,
        "claims",
        FunctionType::get(ctx, {}, tupleOfClaims)
      );

      // build function body
      detached.setInsertionPointToStart(claimsFunc.addEntryBlock());

      // each element position's claim is the impl's where entry at that
      // position
      SmallVector<Value> elements(
          impl.getBody().front().getArguments().drop_front());

      // tuple.make of all claims, and return it
      auto result = MakeOp::create(detached, loc, elements);
      trait::ReturnOp::create(detached, loc, result.getResult());
    }
    allegeRequirements(impl, trait, detached);

    builder.insert(impl);
    return impl;
  }
};

/// The mapper of `mappedTrait` in `module`, declared at `builder`'s insertion
/// point where the module has none: the trait mapping `mappedTrait` across the
/// elements of two tuples, which `MapGenerator` implements, binding the tuple
/// of the resulting claims as its `Claims` associated type:
///
///   trait.trait @tuple.MapEq(%self: !trait.claim<@tuple.MapEq[!S, !O]>) attributes {
///     tuple.impl_generator = "map",
///     tuple.mapped_trait   = @Eq
///   } {
///     trait.assoc_type @Claims
///     trait.method @claims() -> !trait.proj<@tuple.MapEq[!S,!O], "Claims">
///   }
///
/// A mapper is the trait its `tuple.mapped_trait` link makes one
/// (`isMapperOf`), never a trait of some name: the name a declared mapper
/// takes is a label, made unique against the module's symbols, so no other
/// declaration under that name is ever taken for it. The mapper is applied to
/// the two tuple types alone: the tuple of per-element claims is what its impl
/// binds, so an impl for a given pair of tuples determines it rather than a
/// caller having to spell it.
static FlatSymbolRefAttr declareMapperTrait(ModuleOp module,
                                            trait::TraitOp mappedTrait,
                                            OpBuilder &builder) {
  using namespace mlir::trait;
  MLIRContext *ctx = builder.getContext();
  auto mappedRef = FlatSymbolRefAttr::get(ctx, mappedTrait.getSymName());
  for (TraitOp declared : module.getOps<TraitOp>())
    if (isMapperOf(declared, mappedRef))
      return FlatSymbolRefAttr::get(ctx, declared.getSymName());

  SymbolTable symbols(module);
  auto taken = [&](StringRef name) { return symbols.lookup(name) != nullptr; };
  SmallString<32> name("tuple.Map");
  name += mappedTrait.getSymName();
  unsigned suffix = 0;
  if (taken(name))
    name = SymbolTable::generateSymbolName<32>(name, taken, suffix);
  auto mapperRef = FlatSymbolRefAttr::get(ctx, name);

  OpBuilder::InsertionGuard guard(builder);
  Location loc = builder.getUnknownLoc();
  // The two parameters of the trait being built, labelled by their position in
  // its own header: a label is local to the declaration that binds it.
  Type S = trait::PolyType::get(ctx, 0);
  Type O = trait::PolyType::get(ctx, 1);
  auto mapper = TraitOp::create(builder, loc, name, /*typeParams=*/ArrayRef{S, O},
                                /*requirements=*/ArrayRef<Type>{});
  mapper->setAttr("tuple.impl_generator", StringAttr::get(ctx, "map"));
  mapper->setAttr("tuple.mapped_trait", mappedRef);

  builder.setInsertionPointToStart(&mapper.getBody().front());
  AssocTypeOp::create(builder, loc, "Claims", TypeAttr{}, ArrayAttr{});
  auto selfApp = TraitApplicationAttr::get(ctx, mapperRef, ArrayRef{S, O});
  auto claimsTy = builder.getFunctionType(
      /*inputs=*/TypeRange{},
      /*results=*/ProjectionType::get(ctx, selfApp,
                                      StringAttr::get(ctx, "Claims"),
                                      /*assocTypeArgs=*/{}));
  MethodOp::create(builder, loc, "claims", claimsTy);
  return mapperRef;
}

/// Builds the impl of a comparison trait for two tuples of equal arity.
///
/// The impl answers the demanded pair itself and assumes the mapper for the
/// same pair, which it declares where the module lacks it; each method asks the
/// mapper for the elementwise claims and hands them to `tuple.cmp`:
///
///   trait.impl @...(%self: !trait.claim<@PartialEq[tuple<i32>, tuple<i32>]>,
///                   %a: !trait.claim<@tuple.MapPartialEq[tuple<i32>, tuple<i32>]>) {
///     trait.method @eq(%x: tuple<i32>, %y: tuple<i32>) -> i1 {
///       %claims = trait.method.call %a
///         @tuple.MapPartialEq[tuple<i32>, tuple<i32>]::@claims()
///         : () -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
///       %res = tuple.cmp eq @PartialEq, %x, %y, %claims
///       trait.return %res : i1
///     }
///   }
static FailureOr<trait::ImplOp> generateTupleCmpImpl(
    trait::TraitOp trait,
    trait::ClaimType wanted,
    ArrayRef<std::pair<StringRef, CmpPredicate>> methods,
    OpBuilder &builder) {
  using namespace mlir::trait;

  ModuleOp module = trait->getParentOfType<ModuleOp>();
  if (!module)
    return failure();

  // the demanded tuple types are what this impl compares
  auto tuples = groundTuplePair(wanted);
  if (failed(tuples)) return failure();
  auto [lhs, rhs] = *tuples;

  MLIRContext *ctx = builder.getContext();
  Location loc = builder.getUnknownLoc();

  // the mapper trait is what supplies the elementwise claims
  FlatSymbolRefAttr mapperRef = declareMapperTrait(module, trait, builder);

  auto selfApp = TraitApplicationAttr::get(
    ctx,
    FlatSymbolRefAttr::get(ctx, trait.getSymName()),
    ArrayRef<Type>{lhs, rhs}
  );

  // one where entry: the mapper over the same pair of tuples
  auto assumption = TraitApplicationAttr::get(ctx, mapperRef, ArrayRef<Type>{lhs, rhs});

  // the claims the mapper supplies, named as its associated type until the
  // mapper's impl resolves the projection into the claim tuple it binds
  Type claimsTy = ProjectionType::get(ctx, assumption,
                                      StringAttr::get(ctx, "Claims"),
                                      /*assocTypeArgs=*/{});

  // The impl is built whole on a detached builder and inserted once, so a
  // half-built impl is never a candidate.
  OpBuilder detached(ctx);
  ImplOp impl = ImplOp::create(detached, loc, selfApp,
                               ArrayRef<ClaimType>{ClaimType::get(ctx, assumption)});
  Value mapper = impl.getBody().front().getArgument(1);

  for (auto [methodName, predicate] : methods) {
    OpBuilder::InsertionGuard guard(detached);
    detached.setInsertionPoint(impl.getReturn());

    auto fnTy = detached.getFunctionType({lhs, rhs}, detached.getI1Type());
    auto fn = trait::MethodOp::create(detached, loc, methodName, fnTy);

    Block *entry = fn.addEntryBlock();
    detached.setInsertionPointToStart(entry);
    Value self = entry->getArgument(0);
    Value other = entry->getArgument(1);

    // %claims = trait.method.call %a @tuple.Map<Trait>[lhs,rhs]::@claims() : () -> claimsTy
    Value claims = MethodCallOp::create(detached,
      loc,
      /*results=*/TypeRange{claimsTy},
      /*traitName=*/mapperRef.getValue(),
      /*methodName=*/"claims",
      /*claim=*/mapper,
      /*arguments=*/ValueRange{}
    ).getResult(0);

    // %res = tuple.cmp <predicate> @Trait, %self, %other, %claims: the
    // comparison is under the trait this impl serves, the one it names
    Value res = CmpOp::create(detached, loc,
                              CmpPredicateAttr::get(ctx, predicate),
                              FlatSymbolRefAttr::get(ctx, trait.getSymName()),
                              self, other, claims);

    // trait.return %res : i1
    trait::ReturnOp::create(detached, loc, res);
  }
  allegeRequirements(impl, trait, detached);

  builder.insert(impl);
  return impl;
}

/// TuplePartialEqGenerator answers a demanded application of the trait tagged
/// `tuple.impl_generator = "partial_eq"` over two tuples of equal arity with
/// the impl for exactly that pair.
struct TuplePartialEqGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               OpBuilder &builder) const override {
    if (failed(matchGenerator(trait, wanted, "partial_eq", 2, 2)))
      return failure();

    std::pair<StringRef, CmpPredicate> methods[] = {{"eq", CmpPredicate::eq}};
    return generateTupleCmpImpl(trait, wanted, methods, builder);
  }
};

/// TuplePartialOrdGenerator answers a demanded application of the trait tagged
/// `tuple.impl_generator = "partial_ord"` over two tuples of equal arity with
/// the impl for exactly that pair.
struct TuplePartialOrdGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               OpBuilder &builder) const override {
    if (failed(matchGenerator(trait, wanted, "partial_ord", 2, 2)))
      return failure();

    std::pair<StringRef, CmpPredicate> methods[] = {
      {"ge", CmpPredicate::ge},
      {"gt", CmpPredicate::gt},
      {"le", CmpPredicate::le},
      {"lt", CmpPredicate::lt}
    };
    return generateTupleCmpImpl(trait, wanted, methods, builder);
  }
};

void populateImplGenerators(trait::ImplGeneratorSet &generators) {
  generators.add<
    HomogeneousTupleGenerator,
    MapGenerator,
    TuplePartialEqGenerator,
    TuplePartialOrdGenerator,
    TupleGenerator
  >();
}

} // end mlir::tuple
