# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import collections as co
import functools as ft
from collections.abc import Callable, Mapping
from typing import Any, cast, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import sympy as sympy_module


PyTree = Any

concatenate: Callable = sympy_module.Function("concatenate")  # pyright: ignore
stack: Callable = sympy_module.Function("stack")  # pyright: ignore


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)

    return fn_


def _single_args(fn):
    def fn_(*args):
        return fn(args)

    return fn_


_lookup = {
    concatenate: _single_args(jnp.concatenate),
    stack: _single_args(jnp.stack),
    sympy_module.Mul: _reduce(jnp.multiply),
    sympy_module.Add: _reduce(jnp.add),
    sympy_module.div: jnp.divide,
    sympy_module.Abs: jnp.abs,
    sympy_module.sign: jnp.sign,
    sympy_module.ceiling: jnp.ceil,
    sympy_module.floor: jnp.floor,
    sympy_module.log: jnp.log,
    sympy_module.exp: jnp.exp,
    sympy_module.sqrt: jnp.sqrt,
    sympy_module.cos: jnp.cos,
    sympy_module.acos: jnp.arccos,
    sympy_module.sin: jnp.sin,
    sympy_module.asin: jnp.arcsin,
    sympy_module.tan: jnp.tan,
    sympy_module.atan: jnp.arctan,
    sympy_module.atan2: jnp.arctan2,
    sympy_module.cosh: jnp.cosh,
    sympy_module.acosh: jnp.arccosh,
    sympy_module.sinh: jnp.sinh,
    sympy_module.asinh: jnp.arcsinh,
    sympy_module.tanh: jnp.tanh,
    sympy_module.atanh: jnp.arctanh,
    sympy_module.Pow: jnp.power,
    sympy_module.re: jnp.real,
    sympy_module.im: jnp.imag,
    sympy_module.arg: jnp.angle,
    sympy_module.erf: jsp.special.erf,
    sympy_module.Eq: jnp.equal,
    sympy_module.Ne: jnp.not_equal,
    sympy_module.StrictGreaterThan: jnp.greater,
    sympy_module.StrictLessThan: jnp.less,
    sympy_module.LessThan: jnp.less_equal,
    sympy_module.GreaterThan: jnp.greater_equal,
    sympy_module.And: jnp.logical_and,
    sympy_module.Or: jnp.logical_or,
    sympy_module.Not: jnp.logical_not,
    sympy_module.Xor: jnp.logical_xor,
    sympy_module.Max: _reduce(jnp.maximum),
    sympy_module.Min: _reduce(jnp.minimum),
    sympy_module.MatAdd: _reduce(jnp.add),
    sympy_module.Trace: jnp.trace,
    sympy_module.Determinant: jnp.linalg.det,
}

_constant_lookup = {
    sympy_module.E: jnp.e,
    sympy_module.pi: jnp.pi,
    sympy_module.EulerGamma: jnp.euler_gamma,
    sympy_module.I: 1j,
}

_reverse_lookup = {v: k for k, v in _lookup.items()}
assert len(_reverse_lookup) == len(_lookup)


def _item(x):
    if eqx.is_array(x):
        return x.item()
    else:
        return x


class _AbstractNode(eqx.Module):
    @abc.abstractmethod
    def __call__(self, memodict: dict) -> jax.typing.ArrayLike: ...

    @abc.abstractmethod
    def sympy(self, memodict: dict, func_lookup: dict) -> sympy_module.Expr: ...

    # Comparisons based on identity
    __hash__ = object.__hash__
    __eq__ = object.__eq__  # pyright: ignore


class _Symbol(_AbstractNode):
    _name: str

    def __init__(self, expr: sympy_module.Expr):
        self._name = str(expr.name)  # pyright: ignore

    def __call__(self, memodict: dict):
        try:
            return memodict[self._name]
        except KeyError as e:
            raise KeyError(f"Missing input for symbol {self._name}") from e

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy_module.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy_module.Symbol(self._name)


def _maybe_array(val, make_array):
    if make_array:
        return jnp.asarray(val)
    else:
        return val


class _Integer(_AbstractNode):
    _value: jax.typing.ArrayLike

    def __init__(self, expr: sympy_module.Expr, make_array: bool):
        assert isinstance(expr, sympy_module.Integer)
        self._value = _maybe_array(int(expr), make_array)

    def __call__(self, memodict: dict):
        return self._value

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy_module.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy_module.Integer(_item(self._value))


class _Float(_AbstractNode):
    _value: jax.typing.ArrayLike

    def __init__(self, expr: sympy_module.Expr, make_array: bool):
        assert isinstance(expr, sympy_module.Float)
        self._value = _maybe_array(float(expr), make_array)

    def __call__(self, memodict: dict):
        return self._value

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy_module.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy_module.Float(_item(self._value))


class _Rational(_AbstractNode):
    _numerator: jax.typing.ArrayLike
    _denominator: jax.typing.ArrayLike

    def __init__(self, expr: sympy_module.Expr, make_array: bool):
        assert isinstance(expr, sympy_module.Rational)
        numerator = expr.numerator
        denominator = expr.denominator
        if callable(numerator):
            # Support SymPy < 1.10
            numerator = numerator()
        if callable(denominator):
            denominator = denominator()
        self._numerator = _maybe_array(int(numerator), make_array)
        self._denominator = _maybe_array(int(denominator), make_array)

    def __call__(self, memodict: dict):
        return self._numerator / self._denominator

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy_module.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy_module.Integer(_item(self._numerator)) / sympy_module.Integer(
            _item(self._denominator)
        )


class _Constant(_AbstractNode):
    _value: jnp.ndarray
    _expr: sympy_module.Expr

    def __init__(self, expr: sympy_module.Expr, make_array: bool):
        assert expr in _constant_lookup
        self._value = _maybe_array(_constant_lookup[expr], make_array)
        self._expr = expr

    def __call__(self, memodict: dict):
        return self._value

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy_module.Expr:
        return self._expr


class _Func(_AbstractNode):
    _func: Callable
    _args: list

    def __init__(
        self,
        expr: sympy_module.Expr,
        memodict: dict,
        func_lookup: Mapping,
        make_array: bool,
    ):
        try:
            self._func = func_lookup[expr.func]
        except KeyError as e:
            raise KeyError(f"Unsupported Sympy type {type(expr)}") from e
        self._args = [
            _sympy_to_node(
                cast(sympy_module.Expr, arg), memodict, func_lookup, make_array
            )
            for arg in expr.args
        ]

    def __call__(self, memodict: dict):
        args = []
        for arg in self._args:
            try:
                arg_call = memodict[arg]
            except KeyError:
                arg_call = arg(memodict)
                memodict[arg] = arg_call
            args.append(arg_call)
        return self._func(*args)

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy_module.Expr:
        try:
            return memodict[self]
        except KeyError:
            func = func_lookup[self._func]
            args = [arg.sympy(memodict, func_lookup) for arg in self._args]
            out = func(*args)
            memodict[self] = out
            return out


def _sympy_to_node(
    expr: sympy_module.Expr, memodict: dict, func_lookup: Mapping, make_array: bool
) -> _AbstractNode:
    try:
        return memodict[expr]
    except KeyError:
        if isinstance(expr, sympy_module.Symbol):
            out = _Symbol(expr)
        elif isinstance(expr, sympy_module.Integer):
            out = _Integer(expr, make_array)
        elif isinstance(expr, sympy_module.Float):
            out = _Float(expr, make_array)
        elif isinstance(expr, sympy_module.Rational):
            out = _Rational(expr, make_array)
        elif expr in (
            sympy_module.E,
            sympy_module.pi,
            sympy_module.EulerGamma,
            sympy_module.I,
        ):
            out = _Constant(expr, make_array)
        else:
            out = _Func(expr, memodict, func_lookup, make_array)
        memodict[expr] = out
        return out


def _is_node(x):
    return isinstance(x, _AbstractNode)


class SymbolicModule(eqx.Module):
    nodes: PyTree
    has_extra_funcs: bool = eqx.field(static=True)

    def __init__(
        self,
        expressions: PyTree,
        extra_funcs: Optional[dict] = None,
        make_array: bool = True,
    ):
        if extra_funcs is None:
            lookup = _lookup
            self.has_extra_funcs = False
        else:
            lookup = co.ChainMap(extra_funcs, _lookup)
            self.has_extra_funcs = True
        _convert = ft.partial(
            _sympy_to_node,
            memodict=dict(),
            func_lookup=lookup,
            make_array=make_array,
        )
        self.nodes = jtu.tree_map(_convert, expressions)

    def sympy(self) -> sympy_module.Expr:
        if self.has_extra_funcs:
            raise NotImplementedError(
                "SymbolicModule cannot be converted back to SymPy if `extra_funcs` "
                "is passed."
            )
        memodict = dict()
        return jtu.tree_map(
            lambda n: n.sympy(memodict, _reverse_lookup), self.nodes, is_leaf=_is_node
        )

    def __call__(self, **symbols):
        memodict = symbols
        return jtu.tree_map(lambda n: n(memodict), self.nodes, is_leaf=_is_node)
