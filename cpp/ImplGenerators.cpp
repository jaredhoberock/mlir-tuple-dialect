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
    auto noAssumptions = PredicateArrayAttr::get(ctx, ArrayRef<TraitApplicationAttr>{});
    std::string implName = ImplOp::generateSymName(
        claim.getTraitApplication(), noAssumptions);

    // Tuple has no methods or associated types, so the impl body stays empty.
    return ImplOp::create(builder, loc, implName,
                          claim.getTraitApplication(),
                          ArrayRef<TraitApplicationAttr>{});
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
    auto noAssumptions = PredicateArrayAttr::get(ctx, ArrayRef<TraitApplicationAttr>{});
    std::string implName = ImplOp::generateSymName(
        claim.getTraitApplication(), noAssumptions);

    // The impl is built whole on a detached builder and inserted once, so a
    // half-built impl is never a candidate. Its body contains only the
    // Element associated-type binding.
    OpBuilder detached(ctx);
    ImplOp impl = ImplOp::create(detached, loc, implName,
                                 claim.getTraitApplication(),
                                 ArrayRef<TraitApplicationAttr>{});

    detached.setInsertionPointToStart(&impl.getBody().front());
    AssocTypeOp::create(detached, loc, "Element",
                        TypeAttr::get(*elementTy), ArrayAttr{});

    builder.insert(impl);
    return impl;
  }
};

/// MapGenerator synthesizes the `trait.impl` of a "map" trait: the trait that
/// maps another trait across the elements of two tuples and binds the tuple of
/// the resulting claims as its `Claims` associated type.
///
///   trait.trait @tuple.MapEq[!S, !O] attributes {
///     tuple.impl_generator = "map",
///     tuple.mapped_trait   = @Eq
///   } {
///     trait.assoc_type @Claims
///     func.func private @claims() -> !trait.proj<@tuple.MapEq[!S,!O], "Claims">
///   }
///
/// For a wanted claim like:
///
///   !trait.claim<@tuple.MapEq[tuple<i32>, tuple<i64>]>
///
/// this generator answers with the impl for exactly that application:
///
///   trait.impl @... for @tuple.MapEq[tuple<i32>, tuple<i64>]
///     where [@Eq[i32, i64]] {
///       trait.assoc_type @Claims = tuple<!trait.claim<@Eq[i32,i64]>>
///       func.func private @claims() -> tuple<!trait.claim<@Eq[i32,i64]>> {
///         %c0 = trait.assume @Eq[i32,i64]
///         %res = tuple.make(%c0)
///         return %res
///       }
///     }
///
/// Position i of the two tuples contributes the assumption @Eq[Li, Ri], and
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

    // those same applications are the impl's assumptions, one per position
    SmallVector<TraitApplicationAttr> assumptions = llvm::map_to_vector(claims, [](ClaimType c) {
      return c.getTraitApplication();
    });
    auto assumptionsAttr = PredicateArrayAttr::get(ctx, assumptions);

    // the impl answers the demanded application itself
    auto selfApp = TraitApplicationAttr::get(
      ctx,
      FlatSymbolRefAttr::get(ctx, trait.getSymName()),
      ArrayRef<Type>{lhs, rhs}
    );

    // The impl is built whole on a detached builder and inserted once, so a
    // half-built impl is never a candidate.
    OpBuilder detached(ctx);
    ImplOp impl = ImplOp::create(detached,
      loc,
      ImplOp::generateSymName(selfApp, assumptionsAttr),
      selfApp,
      assumptionsAttr
    );

    // the impl's body binds @Claims and defines @claims():
    // - return type: tupleOfClaims
    // - body: %c0 = trait.assume @MappedTrait[L0,R0]
    //         %c1 = trait.assume @MappedTrait[L1,R1]
    //         ...
    //         %res = tuple.make(%c0, %c1, ...)
    //         return %res
    {
      detached.setInsertionPointToStart(&impl.getBody().front());

      AssocTypeOp::create(detached, loc, "Claims",
                          TypeAttr::get(tupleOfClaims), ArrayAttr{});

      // func.func private @claims() -> tupleOfClaims
      auto claimsFunc = func::FuncOp::create(detached,
        loc,
        "claims",
        FunctionType::get(ctx, {}, tupleOfClaims)
      );
      claimsFunc.setPrivate();

      // build function body
      detached.setInsertionPointToStart(claimsFunc.addEntryBlock());

      // emit trait.assume for each element position
      SmallVector<Value> elements = llvm::map_to_vector(claims, [&](ClaimType c) {
        return Value(AssumeOp::create(detached, loc, c));
      });

      // tuple.make of all claims, and return it
      auto result = MakeOp::create(detached, loc, elements);
      func::ReturnOp::create(detached, loc, result.getResult());
    }

    builder.insert(impl);
    return impl;
  }
};

/// Builds the impl of a comparison trait for two tuples of equal arity.
///
/// The impl answers the demanded pair itself and assumes the mapper for the
/// same pair; each method asks the mapper for the elementwise claims and hands
/// them to `tuple.cmp`:
///
///   trait.impl @... for @PartialEq[tuple<i32>, tuple<i32>]
///     where [@tuple.MapPartialEq[tuple<i32>, tuple<i32>]] {
///     func.func private @eq(%self: tuple<i32>, %other: tuple<i32>) -> i1 {
///       %a = trait.assume @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
///       %claims = trait.method.call %a
///         @tuple.MapPartialEq[tuple<i32>, tuple<i32>]::@claims()
///         : () -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
///       %res = tuple.cmp eq, %self, %other, %claims
///       return %res : i1
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

  // the mapper trait must exist: it is what supplies the elementwise claims
  auto mapperRef = FlatSymbolRefAttr::get(ctx, getMapperTraitName(trait.getSymName()));
  if (!SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, mapperRef))
    return failure();

  auto selfApp = TraitApplicationAttr::get(
    ctx,
    FlatSymbolRefAttr::get(ctx, trait.getSymName()),
    ArrayRef<Type>{lhs, rhs}
  );

  // one assumption: the mapper over the same pair of tuples
  auto assumption = TraitApplicationAttr::get(ctx, mapperRef, ArrayRef<Type>{lhs, rhs});
  auto assumptions = PredicateArrayAttr::get(ctx, ArrayRef<TraitApplicationAttr>{assumption});

  // the claims the mapper supplies, named as its associated type until the
  // mapper's impl resolves the projection into the claim tuple it binds
  Type claimsTy = ProjectionType::get(ctx, assumption,
                                      StringAttr::get(ctx, "Claims"),
                                      /*assocTypeArgs=*/{});

  // The impl is built whole on a detached builder and inserted once, so a
  // half-built impl is never a candidate.
  OpBuilder detached(ctx);
  ImplOp impl = ImplOp::create(detached,
    loc,
    ImplOp::generateSymName(selfApp, assumptions),
    selfApp,
    assumptions
  );

  for (auto [methodName, predicate] : methods) {
    OpBuilder::InsertionGuard guard(detached);
    detached.setInsertionPointToEnd(&impl.getBody().front());

    auto fnTy = detached.getFunctionType({lhs, rhs}, detached.getI1Type());
    auto fn = func::FuncOp::create(detached, loc, methodName, fnTy);
    fn.setPrivate();

    Block *entry = fn.addEntryBlock();
    detached.setInsertionPointToStart(entry);
    Value self = entry->getArgument(0);
    Value other = entry->getArgument(1);

    // %a = trait.assume @tuple.Map<Trait>[lhs,rhs]
    Value a = AssumeOp::create(detached, loc, assumption);

    // %claims = trait.method.call %a @tuple.Map<Trait>[lhs,rhs]::@claims() : () -> claimsTy
    Value claims = MethodCallOp::create(detached,
      loc,
      /*results=*/TypeRange{claimsTy},
      /*traitName=*/mapperRef.getValue(),
      /*methodName=*/"claims",
      /*claim=*/a,
      /*arguments=*/ValueRange{}
    ).getResult(0);

    // %res = tuple.cmp <predicate>, %self, %other, %claims
    Value res = CmpOp::create(detached, loc, predicate, self, other, claims);

    // return %res : i1
    func::ReturnOp::create(detached, loc, res);
  }

  builder.insert(impl);
  return impl;
}

/// TuplePartialEqGenerator answers a demanded @PartialEq over two tuples of
/// equal arity with the impl for exactly that pair.
struct TuplePartialEqGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               OpBuilder &builder) const override {
    // only apply to the PartialEq trait
    if (trait.getSymName() != "PartialEq")
      return failure();

    std::pair<StringRef, CmpPredicate> methods[] = {{"eq", CmpPredicate::eq}};
    return generateTupleCmpImpl(trait, wanted, methods, builder);
  }
};

/// TuplePartialOrdGenerator answers a demanded @PartialOrd over two tuples of
/// equal arity with the impl for exactly that pair.
struct TuplePartialOrdGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               OpBuilder &builder) const override {
    // only apply to the PartialOrd trait
    if (trait.getSymName() != "PartialOrd")
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
