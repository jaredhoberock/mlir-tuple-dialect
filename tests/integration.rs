use melior::{
    Context,
    dialect::{arith, func, DialectRegistry},
    ExecutionEngine,
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        Attribute,
        r#type::{FunctionType, IntegerType},
        Block, BlockLike, Location, Module, Operation, Region, RegionLike,
    },
    StringRef,
    pass::{self, PassManager},
    utility::{register_all_dialects},
};
use trait_dialect as trait_;
use tuple_dialect as tuple;
use inline_dialect as inline;

fn append_partial_eq_trait<'c>(
    block: &Block<'c>,
    loc: Location<'c>
) {
    let source = r#"
    trait.trait @PartialEq {
      func.func private @eq(!trait.self, !trait.self) -> i1
    
      func.func private @ne(%self: !trait.self, %other: !trait.self) -> i1 {
        %equal = trait.method.call @PartialEq::@eq<!trait.self>(%self, %other) : (!trait.self, !trait.self) -> i1 to (!trait.self, !trait.self) -> i1
        %true = arith.constant 1 : i1
        %result = arith.xori %equal, %true : i1
        return %result : i1
      }
    }
    "#;

    inline::parse_source_into_block(
        loc,
        &[],
        &[],
        StringRef::new(source),
        block,
    ).expect("valid trait.trait");
}

fn append_partial_eq_i32_impl<'c>(
    block: &Block<'c>,
    loc: Location<'c>,
) {
    let source = r#"
    trait.impl @PartialEq for i32 {
      func.func private @eq(%self: i32, %other: i32) -> i1 {
        %res = arith.cmpi eq, %self, %other : i32
        return %res : i1
      }
    }
    "#;

    inline::parse_source_into_block(
        loc,
        &[],
        &[],
        StringRef::new(source),
        block,
    ).expect("valid trait.impl");
}

fn build_test1_func<'c>(
    ctx: &'c Context,
    loc: Location<'c>
) -> Operation<'c> {
    // build a func.func @test1:
    //
    // func.func @test1() -> i1 {
    //   %a_0 = arith.constant 7 : i32
    //   %b_0 = arith.constant 7 : i32
    //   %a = tuple.constant(%a_0 : i32) : tuple<i32>
    //   %b = tuple.constant(%b_0 : i32) : tuple<i32>
    //   %r = tuple.cmp eq, %a, %b : !str.string
    //   return %r : i1
    // }

    // build the function body
    let region = {
        let block = Block::new(&[]);
        let i32_ty = IntegerType::new(ctx, 32).into();

        let a_0 = block.append_operation(arith::constant(
          ctx,
          IntegerAttribute::new(i32_ty, 7).into(),
          loc,
        )).result(0).unwrap().into();

        let b_0 = block.append_operation(arith::constant(
          ctx,
          IntegerAttribute::new(i32_ty, 7).into(),
          loc,
        )).result(0).unwrap().into();

        let a = block.append_operation(tuple::constant(
            loc,
            &[a_0],
        )).result(0).unwrap().into();

        let b = block.append_operation(tuple::constant(
            loc,
            &[b_0],
        )).result(0).unwrap().into();

        let r = block.append_operation(tuple::cmp(
            loc,
            tuple::CmpPredicate::Eq,
            a,
            b,
        )).result(0).unwrap().into();

        block.append_operation(func::r#return(
            &[r],
            loc,
        ));

        let region = Region::new();
        region.append_block(block);
        region
    };

    // build the function
    let function_type = FunctionType::new(
        &ctx,
        &[],
        &[IntegerType::new(&ctx, 1).into()]
    );

    let mut func_op = func::func(
        &ctx,
        StringAttribute::new(&ctx, "test1"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        loc,
    );
    func_op.set_attribute("llvm.emit_c_interface", Attribute::unit(&ctx));
    func_op
}

fn append_test2_func<'c>(
    block: &Block<'c>,
    loc: Location<'c>,
) {
    // this function applies tuple.cmp on a & b
    // for each different predicate
    // and packs the result of each application into
    // a bit vector
    let source = r#"
    func.func @test2(%a: tuple<i32,i32,i32>, %b: tuple<i32,i32,i32>) -> i64
      attributes { llvm.emit_c_interface } {
      %eq = tuple.cmp eq, %a, %b : tuple<i32,i32,i32>
      %ne = tuple.cmp ne, %a, %b : tuple<i32,i32,i32>

      // XXX TODO implement PartialOrd comparisons
      %lt = arith.constant 0 : i1
      %le = arith.constant 0 : i1
      %gt = arith.constant 0 : i1
      %ge = arith.constant 0 : i1
      //%lt = tuple.cmp lt, %a, %b : tuple<i32,i32,i32>
      //%le = tuple.cmp le, %a, %b : tuple<i32,i32,i32>
      //%gt = tuple.cmp gt, %a, %b : tuple<i32,i32,i32>
      //%ge = tuple.cmp ge, %a, %b : tuple<i32,i32,i32>
    
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c3 = arith.constant 3 : i64
      %c4 = arith.constant 4 : i64
      %c5 = arith.constant 5 : i64
    
      %eq64 = arith.extui %eq : i1 to i64
      %ne64 = arith.extui %ne : i1 to i64
      %lt64 = arith.extui %lt : i1 to i64
      %le64 = arith.extui %le : i1 to i64
      %gt64 = arith.extui %gt : i1 to i64
      %ge64 = arith.extui %ge : i1 to i64
    
      %ne_shifted = arith.shli %ne64, %c1 : i64
      %lt_shifted = arith.shli %lt64, %c2 : i64
      %le_shifted = arith.shli %le64, %c3 : i64
      %gt_shifted = arith.shli %gt64, %c4 : i64
      %ge_shifted = arith.shli %ge64, %c5 : i64
    
      %result1 = arith.ori %eq64, %ne_shifted : i64
      %result2 = arith.ori %result1, %lt_shifted : i64
      %result3 = arith.ori %result2, %le_shifted : i64
      %result4 = arith.ori %result3, %gt_shifted : i64
      %bitmask = arith.ori %result4, %ge_shifted : i64
    
      return %bitmask : i64
    }
    "#;

    inline::parse_source_into_block(
        loc,
        &[],
        &[],
        StringRef::new(source),
        block,
    ).expect("valid func.func");
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct I32x3(i32, i32, i32);

#[test]
fn test_tuple_jit() {
    // create a dialect registry and register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    tuple::register(&context);
    inline::register(&context);

    // make all the dialects available
    context.load_all_available_dialects();

    // create a module
    let loc = Location::unknown(&context);
    let mut module = Module::new(loc);

    // build trait prelude
    append_partial_eq_trait(&module.body(), loc);
    append_partial_eq_i32_impl(&module.body(), loc);

    // build two functions @test1 and @test2
    module.body().append_operation(
        build_test1_func(&context, loc)
    );
    append_test2_func(&module.body(), loc);

    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(trait_::create_monomorphize_pass());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false);

    // test 1
    unsafe {
        let mut result: bool = false;
        let mut packed_args: [*mut (); 1] = [
            &mut result as *mut bool as *mut ()
        ];

        engine.invoke_packed("test1", &mut packed_args)
            .expect("test1 JIT invocation failed");

        assert_eq!(result, true);
    }

    // test 2

    // this helper function helps us actually call the test2 function
    fn call_test2(engine: &ExecutionEngine, a: I32x3, b: I32x3) -> i64 {
        let mut a = a;
        let mut b = b;
        let mut result: i64 = -1;

        // pack pointers to arguments
        let mut packed_args: [*mut (); 3] = [
            &mut a as *mut _ as *mut (),
            &mut b as *mut _ as *mut (),
            &mut result as *mut _ as *mut (),
        ];

        unsafe {
            engine
                .invoke_packed("test2", &mut packed_args)
                .expect("test2 invocation failed");
        }

        result
    }

    // Bit layout of tuple.cmp result (least significant bit on the right):
    //
    //   [ ge gt le lt ne eq ]
    //     5  4  3  2  1  0
    //
    // Each bit is set to 1 if the corresponding predicate evaluates to true.

    // XXX TODO test PartialOrd bits
    //          they are set to be zero right while the PartialOrd predicates are unimplemented

    // true bits: eq
    assert_eq!(call_test2(&engine, I32x3(1,2,3), I32x3(1,2,3)), 0b000001);

    // true bits: ne
    assert_eq!(call_test2(&engine, I32x3(1,2,3), I32x3(1,2,4)), 0b000010);
}
