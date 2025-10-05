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

static std::optional<unsigned> getArityOfFirstTuple(TypeRange types) {
  for (Type ty : types) {
    if (auto tup = dyn_cast<TupleType>(ty)) {
      unsigned a = tup.size();
      return a;
    }
  }
  return std::nullopt;
}

static SmallVector<TupleType> getTupleTypesWithUniquePolymorphicElements(
    MLIRContext* ctx,
    unsigned numTuples,
    unsigned arity) {
  SmallVector<TupleType> result;
  for (int i = 0; i < numTuples; ++i) {
    result.push_back(getTupleTypeWithUniquePolymorphicElements(ctx, arity));
  }
  return result;
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

  return success();
}

static SmallVector<trait::ClaimType> mapTraitAcrossTupleElements(
    MLIRContext *ctx,
    FlatSymbolRefAttr traitRef,
    ArrayRef<TupleType> tupleTypes) {
  using namespace mlir::trait;

  // given tupleTypes := Ti..Tn,
  // where each Ti := tuple<TiE0, TiE1, .. T1iEm>,
  // we want to produce a list of ClaimTypes:
  //
  // !trait.claim<@Trait[T0E0..TnE0]> .. !trait.claim<@Trait[T0Em..TnEm]>

  SmallVector<ClaimType> claims;

  // collect claim i created by applying the trait
  // to element i of each tuple in order
  unsigned m = tupleTypes.front().size();
  for (unsigned i = 0; i < m; ++i) {
    // get element i from each tuple
    SmallVector<Type> elements;
    for (auto tup : tupleTypes) {
      elements.push_back(tup.getType(i));
    }

    // apply the trait to these elements 
    claims.push_back(ClaimType::get(ctx, traitRef, elements));
  }

  return claims;
}

/// MapGenerator synthesizes `trait.impl`s for traits that declare themselves
/// as "map" generators over tuple arguments.
///
/// A "map" generator allows you to define a trait that, given one or more
/// tuple type arguments, produces a new tuple of per-element claims by
/// applying some other trait elementwise. For example:
///
///   trait.trait @tuple.MapEq[!T, !R] attributes {
///     tuple.impl_generator = "map",
///     tuple.mapped_trait   = @Eq
///   } {
///     func.func private @claims() -> !R
///   }
///
/// For a wanted claim like:
///
///   !trait.claim<@tuple.MapEq[tuple<i32>, tuple<!trait.claim<@Eq[i32]>>]>
///
/// this generator synthesizes an impl of the form:
///
///   trait.impl @tuple.MapEq_impl_arity1
///     for @tuple.MapEq[tuple<!trait.poly<0>>,
///                      tuple<!trait.claim<@Eq[!trait.poly<0>]>>]
///     where [@Eq[!trait.poly<0>]] {
///       func.func private @claims() -> tuple<!trait.claim<@Eq[!trait.poly<0>]>> {
///         %c0 = trait.assume @Eq[!trait.poly<0>]
///         %res = tuple.make(%c0)
///         return %res
///       }
///     }
///
/// More generally:
///   - The leading type arguments must each be either a TupleType or a
///     type variable. All TupleType arguments must have a consistent arity N.
///   - For each element position i in 0..N-1, an assumption
///       @mapped_trait[T0Ei, T1Ei, ..., TkEi]
///     is generated, where each Ti is one of the leading arguments.
///   - The final type argument of the self-claim is a tuple of all those
///     elementwise claims.
///   - The synthesized impl’s @claims method body builds and returns that
///     tuple by `trait.assume`ing each per-element claim.
struct MapGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               PatternRewriter &rewriter) const override {
    using namespace mlir::trait;

    // trait must opt into this generator and have at least 2 type args
    // XXX TODO there's no need to actually check the number of args here
    //          because substituteWith will check that for us below
    if (failed(matchGenerator(trait, wanted, "map", 2)))
      return failure();

    // the tuple.mapped_trait attribute must exist
    auto mappedTrait = trait->getAttrOfType<FlatSymbolRefAttr>("tuple.mapped_trait");
    if (!mappedTrait) return failure();

    // get the arity of the first TupleType in the wanted claim's type args
    auto arity = getArityOfFirstTuple(wanted.getTraitApplication().getTypeArgs());
    if (!arity) return failure();

    // k: the number of tuple args
    int k = wanted.getTraitApplication().getTypeArgs().size() - 1;

    // create k fresh polymorphic tuple types of the requested arity n:
    // Ti := tuple<TiE1..TiEn>
    // where each TiEj is a unique !trait.poly
    MLIRContext *ctx = trait.getContext();
    SmallVector<TupleType> polyTupleTypes = getTupleTypesWithUniquePolymorphicElements(ctx, k, *arity);

    // XXX TODO if we don't end up generating an impl, then we've "wasted" the unique args here
    //          consider building a guard that reclaims the unused unique IDs

    // create a TupleType representing the trait mapped across the elements of these tuples:
    // tuple<!trait.claim<@MappedTrait[T1E1..TkE1]>..!trait.claim<@MappedTrait[T1En..TkEn]>>
    SmallVector<ClaimType> claims = mapTraitAcrossTupleElements(ctx, mappedTrait, polyTupleTypes);
    TupleType tupleOfClaims = TupleType::get(ctx, toTypes(claims));

    // the type arguments of our impl's claim are
    // polyTupleTypes followed by tupleOfClaims
    SmallVector<Type> ourTypeArgs = toTypes(polyTupleTypes);
    ourTypeArgs.push_back(tupleOfClaims);

    // build the claim of the impl we can generate:
    // !trait.claim<@Trait[polyTupleTy1..polyTupleTyK, tupleOfClaims]>
    auto ourTraitRef = FlatSymbolRefAttr::get(ctx, trait.getSymName());
    auto ourClaim = ClaimType::get(ctx, ourTraitRef, ourTypeArgs);

    // get the module for the following
    ModuleOp module = trait->getParentOfType<ModuleOp>();
    if (!module) return failure();

    // check that the wanted claim can unify with our formal claim
    if (failed(buildSpecializationSubstitution(ourClaim, wanted, module)))
      return failure();

    // build assumptions: one @MappedTrait[...] per element position
    SmallVector<TraitApplicationAttr> assumptions = llvm::map_to_vector(claims, [](ClaimType c) {
      return c.getTraitApplication();
    });

    // synthesize the trait.impl
    auto loc = rewriter.getUnknownLoc();
    auto name = (trait.getSymName() + Twine("_impl_arity") + Twine(*arity)).str();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    ImplOp impl = rewriter.create<ImplOp>(
      loc,
      StringAttr::get(ctx, name),
      ourClaim.getTraitApplication(),
      TraitApplicationArrayAttr::get(ctx, assumptions)
    );

    // define the @claims method body:
    // - return type: tupleOfClaims
    // - body: %c0 = trait.assume @MappedTrait[...]
    //         %c1 = trait.assume @MappedTrait[...]
    //         ...
    //         %res = tuple.make(%c0, %c1, ...)
    //         return %res
    {
      Block &implBody = impl.getBody().front();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&implBody);

      // func.func private @claims() -> tupleOfClaims
      FunctionType claimsFnTy = FunctionType::get(ctx, {}, tupleOfClaims);
      auto claimsFunc = rewriter.create<func::FuncOp>(
        loc,
        "claims",
        claimsFnTy
      );
      claimsFunc.setPrivate();

      // build function body
      Block *entry = claimsFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(entry);

      // emit trait.assume for each type in the tuple of claims type
      SmallVector<Value> elements;
      for (ClaimType c : claims) {
        auto assume = rewriter.create<AssumeOp>(loc, c);
        elements.push_back(assume.getResult());
      }

      // tuple.make of all claims
      auto result = rewriter.create<MakeOp>(loc, elements);

      // return
      rewriter.create<func::ReturnOp>(loc, result.getResult());
    }

    return impl;
  }
};

/// TuplePartialEqGenerator synthesizes a polymorphic `trait.impl` of PartialEq for tuples
struct TuplePartialEqGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               PatternRewriter &rewriter) const override {
    using namespace trait;

    // generate the following impl if 
    // 1. it does not already exist, and
    // 2. the @tuple.MapPartialEq trait does exist:
    //
    // !S = !tuple.poly<unique>
    // !O = !tuple.poly<unique>
    // !C = !tuple.poly<unique>
    // trait.impl @tuple.PartialEq for @PartialEq[!S,!O] where [
    //   @tuple.MapPartialEq[!S,!O,!C]
    // ] {
    //   func.func private @eq(%self: !S, %other: !O) -> i1 {
    //     %a = trait.assume @tuple.MapPartialEq[!S,!O]
    //     %claims = trait.method.call %a @tuple.MapPartialEq[!S,!O,!C]::@claims()
    //       : () -> !C
    //     %res = tuple.cmp eq, %self, %other, %claims : !S, !O, !C
    //     return %res : i1
    //   }
    // }

    // only apply to the PartialEq trait
    if (trait.getSymName() != "PartialEq")
      return failure();

    ModuleOp module = trait->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    MLIRContext *ctx = rewriter.getContext();

    // if the impl "tuple.PartialEq" already exists, do nothing
    StringRef implName = "tuple.PartialEq";
    if (SymbolTable::lookupNearestSymbolFrom<ImplOp>(module, FlatSymbolRefAttr::get(ctx, implName)))
      return failure();

    // the mapper trait must exist in order to generate the impl
    auto mapPartialEqRef = FlatSymbolRefAttr::get(ctx, "tuple.MapPartialEq");
    if (!SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, mapPartialEqRef))
      return failure();

    // build polymorphic tuple parameters
    auto S = tuple::PolyType::getUnique(ctx);
    auto O = tuple::PolyType::getUnique(ctx);
    auto C = tuple::PolyType::getUnique(ctx);

    // our impl's claim: !trait.claim<PartialEq[!S,!O]>
    auto partialEqRef = FlatSymbolRefAttr::get(ctx, trait.getSymName());
    auto ourClaim = ClaimType::get(ctx, partialEqRef, {S,O});

    // one assumption: @tuple.MapPartialEq[!S,!O,!C]
    auto assumption = TraitApplicationAttr::get(ctx, mapPartialEqRef, {S,O,C});
    auto assumptions = TraitApplicationArrayAttr::get(ctx, {assumption});

    // create impl
    Location loc = rewriter.getUnknownLoc();
    auto impl = rewriter.create<ImplOp>(
      loc,
      implName,
      ourClaim.getTraitApplication(),
      assumptions
    );

    // define method: func private @eq(%self: !S, %other: !O) -> i1
    {
      Block &body = impl.getBody().front();
      rewriter.setInsertionPointToStart(&body);

      auto i1 = rewriter.getI1Type();
      auto eqTy = rewriter.getFunctionType({S,O}, i1);
      auto eqFn = rewriter.create<func::FuncOp>(loc, "eq", eqTy);
      eqFn.setPrivate();

      Block *entry = eqFn.addEntryBlock();
      rewriter.setInsertionPointToStart(entry);
      Value self = entry->getArgument(0);
      Value other = entry->getArgument(1);

      // %a = trait.assume @tuple.MapPartialEq[!S,!O,!C]
      Value a = rewriter.create<AssumeOp>(loc, assumption);

      // %claims = trait.method.call %a @tuple.MapPartialEq[!S,!O,!C]::@claims() : () -> !C
      Value claims = rewriter.create<MethodCallOp>(
        loc,
        /*results=*/TypeRange{C},
        /*traitName=*/"tuple.MapPartialEq",
        /*methodName=*/"claims",
        /*claim=*/a,
        /*arguments=*/ValueRange{}
      ).getResult(0);

      // %res = tuple.cmp eq, %self, %other, %claims : !S, !O, !C
      Value res = rewriter.create<CmpOp>(
        loc,
        CmpPredicate::eq,
        self, other, claims
      );

      // return %res : i1
      rewriter.create<func::ReturnOp>(loc, res);
    }

    return impl;
  }
};

/// TuplePartialOrdGenerator synthesizes a polymorphic `trait.impl` of
/// PartialOrd for tuples:
///
///   !S = !tuple.poly<unique>
///   !O = !tuple.poly<unique>
///   !C = !tuple.poly<unique>
///   trait.impl @tuple.PartialOrd for @PartialOrd[!S,!O] where [
///     @tuple.MapPartialOrd[!S,!O,!C]
///   ] {
///     func.func private @ge(%self: !S, %other: !O) -> i1 { ... tuple.cmp ge ... }
///     func.func private @gt(%self: !S, %other: !O) -> i1 { ... tuple.cmp gt ... }
///     func.func private @le(%self: !S, %other: !O) -> i1 { ... tuple.cmp le ... }
///     func.func private @lt(%self: !S, %other: !O) -> i1 { ... tuple.cmp lt ... }
///   }
struct TuplePartialOrdGenerator : trait::ImplGenerator {
  FailureOr<trait::ImplOp>
  generateImpl(trait::TraitOp trait,
               trait::ClaimType wanted,
               PatternRewriter &rewriter) const override {
    using namespace trait;

    // only for PartialOrd
    if (trait.getSymName() != "PartialOrd")
      return failure();

    ModuleOp module = trait->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    MLIRContext* ctx = rewriter.getContext();

    // if the impl already exists, do nothing
    StringRef implName = "tuple.PartialOrd";
    if (SymbolTable::lookupNearestSymbolFrom<ImplOp>(
          module, FlatSymbolRefAttr::get(ctx, implName)))
      return failure();

    // require the mapper trait to exist
    auto mapRef = FlatSymbolRefAttr::get(ctx, "tuple.MapPartialOrd");
    if (!SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, mapRef))
      return failure();

    // polymorphic tuple type parameters
    auto S = tuple::PolyType::getUnique(ctx);
    auto O = tuple::PolyType::getUnique(ctx);
    auto C = tuple::PolyType::getUnique(ctx);

    // our impl's self claim: !trait.claim<@PartialOrd[!S,!O]>
    auto partialOrdRef = FlatSymbolRefAttr::get(ctx, trait.getSymName());
    auto ourClaim = ClaimType::get(ctx, partialOrdRef, {S, O});

    // one assumption on the mapper: @tuple.MapPartialOrd[!S,!O,!C]
    auto assumption = TraitApplicationAttr::get(ctx, mapRef, {S, O, C});
    auto assumptions = TraitApplicationArrayAttr::get(ctx, {assumption});

    // create the impl op
    Location loc = rewriter.getUnknownLoc();
    auto impl = rewriter.create<ImplOp>(
      loc,
      implName,
      ourClaim.getTraitApplication(),
      assumptions
    );

    // helper to define one of lt/le/gt/ge with tuple.cmp + mapped claims
    auto buildCmpMethod = [&](StringRef methodName, tuple::CmpPredicate pred) {
      PatternRewriter::InsertionGuard guard(rewriter);

      Block &body = impl.getBody().front();
      rewriter.setInsertionPointToStart(&body);

      auto i1 = rewriter.getI1Type();
      auto fnTy = rewriter.getFunctionType({S,O}, i1);
      auto fn = rewriter.create<func::FuncOp>(loc, methodName, fnTy);
      fn.setPrivate();

      Block *entry = fn.addEntryBlock();
      rewriter.setInsertionPointToStart(entry);
      Value self = entry->getArgument(0);
      Value other = entry->getArgument(1);

      // %a = trait.assume @tuple.MapPartialOrd[!S,!O,!C]
      Value a = rewriter.create<AssumeOp>(loc, assumption);

      // %claims = trait.method.call %a @tuple.MapPartialOrd[!S,!O,!C]::@claims() : () -> !C
      Value claims = rewriter.create<MethodCallOp>(
        loc,
        /*results=*/TypeRange{C},
        /*traitName=*/"tuple.MapPartialOrd",
        /*methodName=*/"claims",
        /*claim=*/a,
        /*arguments=*/ValueRange{}
      ).getResult(0);

      // %res = tuple.cmp <pred>, %self, %other, %claims : !S, !O, !C
      Value res = rewriter.create<CmpOp>(loc, pred, self, other, claims);

      // return %res : i1
      rewriter.create<func::ReturnOp>(loc, res);
    };

    // define all four methods
    buildCmpMethod("ge", tuple::CmpPredicate::ge);
    buildCmpMethod("gt", tuple::CmpPredicate::gt);
    buildCmpMethod("le", tuple::CmpPredicate::le);
    buildCmpMethod("lt", tuple::CmpPredicate::lt);

    return impl;
  }
};

void populateImplGenerators(trait::ImplGeneratorSet &generators) {
  generators.add<
    MapGenerator,
    TuplePartialEqGenerator,
    TuplePartialOrdGenerator
  >();
}

} // end mlir::tuple
