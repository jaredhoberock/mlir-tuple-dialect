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

    // Tuple has no members or associated types of its own; the impl leaves
    // every member to its trait's default and its body alleges whatever the
    // trait requires, for selection to prove where a use reads it.
    ImplOp impl = ImplOp::create(builder, loc, claim.getTraitApplication(),
                                 ArrayRef<ClaimType>{});
    impl.completeMembers(trait);
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
    // half-built impl is never a candidate. It binds Element, leaves every
    // member to its trait's default, and alleges what the trait requires, for
    // selection to prove where a use reads it.
    OpBuilder detached(ctx);
    ImplOp impl = ImplOp::create(detached, loc, claim.getTraitApplication(),
                                 ArrayRef<ClaimType>{});
    impl.bindAssociatedType("Element", *elementTy);
    impl.completeMembers(trait);
    allegeRequirements(impl, trait, detached);

    builder.insert(impl);
    return impl;
  }
};

/// MapGenerator synthesizes the `trait.impl` of a "map" trait: the trait that
/// maps another trait across the elements of two tuples and binds the tuple of
/// the resulting claims as its `Claims` associated type.
///
///   trait.trait @tuple.MapEq(!trait.claim<@tuple.MapEq[!S, !O]>)
///       assoc {Claims = 0} members [@tuple.MapEq.claims] attributes {
///     tuple.impl_generator = "map",
///     tuple.mapped_trait   = @Eq
///   }
///   func.func private @tuple.MapEq.claims(!trait.claim<@tuple.MapEq[!S, !O]>)
///       -> !trait.proj<@tuple.MapEq[!S,!O], "Claims">
///
/// For a wanted claim like:
///
///   !trait.claim<@tuple.MapEq[tuple<i32>, tuple<i64>]>
///
/// this generator answers with the impl for exactly that application:
///
///   trait.impl @I(%self: !trait.claim<@tuple.MapEq[tuple<i32>, tuple<i64>]>,
///                 %eq: !trait.claim<@Eq[i32, i64]>)
///       bindings {Claims = tuple<!trait.claim<@Eq[i32,i64]>>}
///       members [@tuple.MapEq.claims -> @I.claims] {
///   }
///   func.func private @I.claims(%self: ..., %eq: !trait.claim<@Eq[i32, i64]>)
///       -> tuple<!trait.claim<@Eq[i32,i64]>> {
///     %res = tuple.make(%eq)
///     return %res
///   }
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

    // The mapper's one member, which the impl implements.
    FlatSymbolRefAttr claimsMember = trait.getMemberNamed("claims");
    if (!claimsMember) return failure();

    // The impl is built whole on a detached builder and inserted once, so a
    // half-built impl is never a candidate. Its where entries are those same
    // applications, one per position, and it binds @Claims to their tuple.
    OpBuilder detached(ctx);
    ImplOp impl = ImplOp::create(detached, loc, selfApp, claims);
    impl.bindAssociatedType("Claims", tupleOfClaims);
    allegeRequirements(impl, trait, detached);
    builder.insert(impl);

    // The function implementing @claims returns the tuple of the impl's where
    // entries, the parameters after the leading self claim:
    //   %res = tuple.make(%c0, %c1, ...)
    //   return %res
    func::FuncOp claimsFunc = impl.addMember(
        builder, loc, claimsMember, FunctionType::get(ctx, {}, tupleOfClaims));
    {
      OpBuilder::InsertionGuard guard(builder);
      Block &entry = claimsFunc.getBody().front();
      builder.setInsertionPointToStart(&entry);
      SmallVector<Value> elements(entry.getArguments().drop_front());
      auto result = MakeOp::create(builder, loc, elements);
      func::ReturnOp::create(builder, loc, result.getResult());
    }
    impl.completeMembers(trait);
    return impl;
  }
};

/// The mapper of `mappedTrait` in `module`, declared at `builder`'s insertion
/// point where the module has none: the trait mapping `mappedTrait` across the
/// elements of two tuples, which `MapGenerator` implements, binding the tuple
/// of the resulting claims as its `Claims` associated type:
///
///   trait.trait @tuple.MapEq(!trait.claim<@tuple.MapEq[!S, !O]>)
///       assoc {Claims = 0} members [@tuple.MapEq.claims] attributes {
///     tuple.impl_generator = "map",
///     tuple.mapped_trait   = @Eq
///   }
///   func.func private @tuple.MapEq.claims(!trait.claim<@tuple.MapEq[!S, !O]>)
///       -> !trait.proj<@tuple.MapEq[!S,!O], "Claims">
///
/// A mapper is the trait its `tuple.mapped_trait` link makes one
/// (`isMapperOf`), never a trait of some name: the name a declared mapper
/// takes is a label, made unique against the module's symbols, so no other
/// declaration under that name is ever taken for it. The mapper is applied to
/// the two tuple types alone: the tuple of per-element claims is what its impl
/// binds, so an impl for a given pair of tuples determines it rather than a
/// caller having to spell it.
static trait::TraitOp declareMapperTrait(ModuleOp module,
                                        trait::TraitOp mappedTrait,
                                        OpBuilder &builder) {
  using namespace mlir::trait;
  MLIRContext *ctx = builder.getContext();
  auto mappedRef = FlatSymbolRefAttr::get(ctx, mappedTrait.getSymName());
  for (TraitOp declared : module.getOps<TraitOp>())
    if (isMapperOf(declared, mappedRef))
      return declared;

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

  mapper.declareAssociatedType("Claims", 0);
  auto selfApp = TraitApplicationAttr::get(ctx, mapperRef, ArrayRef{S, O});
  auto claimsTy = builder.getFunctionType(
      /*inputs=*/TypeRange{},
      /*results=*/ProjectionType::get(ctx, selfApp,
                                      StringAttr::get(ctx, "Claims"),
                                      /*assocTypeArgs=*/{}));
  builder.setInsertionPointAfter(mapper);
  mapper.addMember(builder, loc, "claims", claimsTy, /*withDefault=*/false);
  return mapper;
}

/// Builds the impl of a comparison trait for two tuples of equal arity.
///
/// The impl answers the demanded pair itself and assumes the mapper for the
/// same pair, which it declares where the module lacks it; the function
/// implementing each member asks the mapper for the elementwise claims and
/// hands them to `tuple.cmp`:
///
///   trait.impl @I(%self: !trait.claim<@PartialEq[tuple<i32>, tuple<i32>]>,
///                 %a: !trait.claim<@tuple.MapPartialEq[tuple<i32>, tuple<i32>]>)
///       members [@PartialEq.eq -> @I.eq, @PartialEq.ne -> @PartialEq.ne] {
///   }
///   func.func private @I.eq(%self: ..., %a: ..., %x: tuple<i32>, %y: tuple<i32>) -> i1 {
///     %claims = trait.func.call @tuple.MapPartialEq.claims(%a)
///       : (!trait.claim<@tuple.MapPartialEq[tuple<i32>, tuple<i32>]>)
///         -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
///     %res = tuple.cmp eq @PartialEq, %x, %y, %claims
///     return %res : i1
///   }
///
/// A trait declaring no member of one of `methods`' names gets no impl.
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

  // Each member the impl implements, by the name the generator's tag promises.
  SmallVector<FlatSymbolRefAttr> members;
  for (auto [methodName, predicate] : methods) {
    members.push_back(trait.getMemberNamed(methodName));
    if (!members.back())
      return failure();
  }

  // the mapper trait is what supplies the elementwise claims, through its one
  // member
  TraitOp mapper = declareMapperTrait(module, trait, builder);
  auto mapperRef = FlatSymbolRefAttr::get(ctx, mapper.getSymName());
  FlatSymbolRefAttr claimsMember = mapper.getMemberNamed("claims");
  if (!claimsMember)
    return failure();

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
  allegeRequirements(impl, trait, detached);
  builder.insert(impl);

  for (auto [member, method] : llvm::zip(members, methods)) {
    auto fnTy = builder.getFunctionType({lhs, rhs}, builder.getI1Type());
    func::FuncOp fn = impl.addMember(builder, loc, member, fnTy);

    // The function leads with the impl's self claim and its where entry, the
    // mapper's claim, then the two tuples it compares.
    OpBuilder::InsertionGuard guard(builder);
    Block &entry = fn.getBody().front();
    builder.setInsertionPointToStart(&entry);
    Value mapperClaim = entry.getArgument(1);
    Value self = entry.getArgument(2);
    Value other = entry.getArgument(3);

    // %claims = trait.func.call @<mapper>.claims(%a)
    //   : (!trait.claim<@tuple.Map<Trait>[lhs,rhs]>) -> claimsTy
    Value claims = FuncCallOp::create(builder, loc, TypeRange{claimsTy},
                                      claimsMember, ValueRange{mapperClaim})
                       .getResult(0);

    // %res = tuple.cmp <predicate> @Trait, %self, %other, %claims: the
    // comparison is under the trait this impl serves, the one it names
    Value res = CmpOp::create(builder, loc,
                              CmpPredicateAttr::get(ctx, method.second),
                              FlatSymbolRefAttr::get(ctx, trait.getSymName()),
                              self, other, claims);
    func::ReturnOp::create(builder, loc, res);
  }
  impl.completeMembers(trait);
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
