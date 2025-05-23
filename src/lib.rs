use melior::{ir::{Location, Value, ValueLike, Operation}, Context};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirValue};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum CmpPredicate {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

#[link(name = "tuple_dialect")]
unsafe extern "C" {
    fn tupleRegisterDialect(ctx: MlirContext);
    fn tupleConstantOpCreate(loc: MlirLocation, elements: *const MlirValue, n: isize) -> MlirOperation;
    fn tupleGetOpCreate(loc: MlirLocation, tuple: MlirValue, index: isize) -> MlirOperation;
    fn tupleCmpOpCreate(loc: MlirLocation, predicate: CmpPredicate, lhs: MlirValue, rhs: MlirValue) -> MlirOperation;
}

pub fn register(context: &Context) {
    unsafe { tupleRegisterDialect(context.to_raw()) }
}

pub fn constant<'c>(loc: Location<'c>, values: &[Value<'c, '_>]) -> Operation<'c> {
    let op = unsafe {
        tupleConstantOpCreate(loc.to_raw(), values.as_ptr() as *const _, values.len() as isize)
    };
    unsafe { Operation::from_raw(op) }
}

pub fn get<'c>(loc: Location<'c>, tuple: Value<'c, '_>, index: isize) -> Operation<'c> {
    let op = unsafe {
        tupleGetOpCreate(loc.to_raw(), tuple.to_raw(), index)
    };
    unsafe { Operation::from_raw(op) }
}

pub fn cmp<'c>(
    loc: Location<'c>,
    pred: CmpPredicate,
    lhs: Value<'c,'_>,
    rhs: Value<'c,'_>,
) -> Operation<'c> {
    unsafe {
        Operation::from_raw(tupleCmpOpCreate(
            loc.to_raw(),
            pred,
            lhs.to_raw(),
            rhs.to_raw(),
        ))
    }
}
