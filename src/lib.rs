// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use melior::{ir::{Location, Value, ValueLike, Operation}, Context};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirStringRef, MlirValue};

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
    fn tupleCmpOpCreate(loc: MlirLocation, predicate: CmpPredicate, trait_: MlirStringRef, lhs: MlirValue, rhs: MlirValue, claims: MlirValue) -> MlirOperation;
    fn tupleGetOpCreate(loc: MlirLocation, tuple: MlirValue, index: isize) -> MlirOperation;
    fn tupleMakeOpCreate(loc: MlirLocation, elements: *const MlirValue, n: isize) -> MlirOperation;
}

pub fn register(context: &Context) {
    unsafe { tupleRegisterDialect(context.to_raw()) }
}

/// `tuple.cmp` of `lhs` and `rhs` by `pred`, under the trait named `trait_`.
pub fn cmp<'c>(
    loc: Location<'c>,
    pred: CmpPredicate,
    trait_: &str,
    lhs: Value<'c,'_>,
    rhs: Value<'c,'_>,
    claims: Option<Value<'c,'_>>,
) -> Operation<'c> {
    let trait_ = MlirStringRef { data: trait_.as_ptr() as *const _, length: trait_.len() };
    let claims = claims.map_or(MlirValue { ptr: std::ptr::null_mut() }, |claims| claims.to_raw());
    unsafe {
        Operation::from_raw(tupleCmpOpCreate(
            loc.to_raw(),
            pred,
            trait_,
            lhs.to_raw(),
            rhs.to_raw(),
            claims,
        ))
    }
}

pub fn get<'c>(loc: Location<'c>, tuple: Value<'c, '_>, index: isize) -> Operation<'c> {
    let op = unsafe {
        tupleGetOpCreate(loc.to_raw(), tuple.to_raw(), index)
    };
    unsafe { Operation::from_raw(op) }
}

pub fn make<'c>(loc: Location<'c>, values: &[Value<'c, '_>]) -> Operation<'c> {
    let op = unsafe {
        tupleMakeOpCreate(loc.to_raw(), values.as_ptr() as *const _, values.len() as isize)
    };
    unsafe { Operation::from_raw(op) }
}
