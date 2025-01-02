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
import sympy


PyTree = Any

concatenate: Callable = sympy.Function("concatenate")  # pyright: ignore
stack: Callable = sympy.Function("stack")  # pyright: ignore


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
    sympy.Mul: _reduce(jnp.multiply),
    sympy.Add: _reduce(jnp.add),
    sympy.div: jnp.divide,
    sympy.Abs: jnp.abs,
    sympy.sign: jnp.sign,
    sympy.ceiling: jnp.ceil,
    sympy.floor: jnp.floor,
    sympy.log: jnp.log,
    sympy.exp: jnp.exp,
    sympy.sqrt: jnp.sqrt,
    sympy.cos: jnp.cos,
    sympy.acos: jnp.arccos,
    sympy.sin: jnp.sin,
    sympy.asin: jnp.arcsin,
    sympy.tan: jnp.tan,
    sympy.atan: jnp.arctan,
    sympy.atan2: jnp.arctan2,
    sympy.cosh: jnp.cosh,
    sympy.acosh: jnp.arccosh,
    sympy.sinh: jnp.sinh,
    sympy.asinh: jnp.arcsinh,
    sympy.tanh: jnp.tanh,
    sympy.atanh: jnp.arctanh,
    sympy.Pow: jnp.power,
    sympy.re: jnp.real,
    sympy.im: jnp.imag,
    sympy.arg: jnp.angle,
    sympy.erf: jsp.special.erf,
    sympy.Eq: jnp.equal,
    sympy.Ne: jnp.not_equal,
    sympy.StrictGreaterThan: jnp.greater,
    sympy.StrictLessThan: jnp.less,
    sympy.LessThan: jnp.less_equal,
    sympy.GreaterThan: jnp.greater_equal,
    sympy.And: jnp.logical_and,
    sympy.Or: jnp.logical_or,
    sympy.Not: jnp.logical_not,
    sympy.Xor: jnp.logical_xor,
    sympy.Max: _reduce(jnp.maximum),
    sympy.Min: _reduce(jnp.minimum),
    sympy.MatAdd: _reduce(jnp.add),
    sympy.Trace: jnp.trace,
    sympy.Determinant: jnp.linalg.det,
}

_constant_lookup = {
    sympy.E: jnp.e,
    sympy.pi: jnp.pi,
    sympy.EulerGamma: jnp.euler_gamma,
    sympy.I: 1j,
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
    def __call__(self, memodict: dict) -> jax.typing.ArrayLike:
        ...

    @abc.abstractmethod
    def sympy(self, memodict: dict, func_lookup: dict) -> sympy.Expr:
        ...

    # Comparisons based on identity
    __hash__ = object.__hash__
    __eq__ = object.__eq__  # pyright: ignore


class _Symbol(_AbstractNode):
    _name: str

    def __init__(self, expr: sympy.Expr):
        self._name = str(expr.name)  # pyright: ignore

    def __call__(self, memodict: dict):
        try:
            return memodict[self._name]
        except KeyError as e:
            raise KeyError(f"Missing input for symbol {self._name}") from e

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Symbol(self._name)


def _maybe_array(val, make_array):
    if make_array:
        return jnp.asarray(val)
    else:
        return val


class _Integer(_AbstractNode):
    _value: jax.typing.ArrayLike

    def __init__(self, expr: sympy.Expr, make_array: bool):
        assert isinstance(expr, sympy.Integer)
        self._value = _maybe_array(int(expr), make_array)

    def __call__(self, memodict: dict):
        return self._value

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Integer(_item(self._value))


class _Float(_AbstractNode):
    _value: jax.typing.ArrayLike

    def __init__(self, expr: sympy.Expr, make_array: bool):
        assert isinstance(expr, sympy.Float)
        self._value = _maybe_array(float(expr), make_array)

    def __call__(self, memodict: dict):
        return self._value

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Float(_item(self._value))


class _Rational(_AbstractNode):
    _numerator: jax.typing.ArrayLike
    _denominator: jax.typing.ArrayLike

    def __init__(self, expr: sympy.Expr, make_array: bool):
        assert isinstance(expr, sympy.Rational)
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

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Integer(_item(self._numerator)) / sympy.Integer(
            _item(self._denominator)
        )


class _Constant(_AbstractNode):
    _value: jnp.ndarray
    _expr: sympy.Expr

    def __init__(self, expr: sympy.Expr, make_array: bool):
        assert expr in _constant_lookup
        self._value = _maybe_array(_constant_lookup[expr], make_array)
        self._expr = expr

    def __call__(self, memodict: dict):
        return self._value

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy.Expr:
        return self._expr


class _Func(_AbstractNode):
    _func: Callable
    _args: list

    def __init__(
        self, expr: sympy.Expr, memodict: dict, func_lookup: Mapping, make_array: bool
    ):
        try:
            self._func = func_lookup[expr.func]
        except KeyError as e:
            raise KeyError(f"Unsupported Sympy type {type(expr)}") from e
        self._args = [
            _sympy_to_node(cast(sympy.Expr, arg), memodict, func_lookup, make_array)
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

    def sympy(self, memodict: dict, func_lookup: dict) -> sympy.Expr:
        try:
            return memodict[self]
        except KeyError:
            func = func_lookup[self._func]
            args = [arg.sympy(memodict, func_lookup) for arg in self._args]
            out = func(*args)
            memodict[self] = out
            return out


def _sympy_to_node(
    expr: sympy.Expr, memodict: dict, func_lookup: Mapping, make_array: bool
) -> _AbstractNode:
    try:
        return memodict[expr]
    except KeyError:
        if isinstance(expr, sympy.Symbol):
            out = _Symbol(expr)
        elif isinstance(expr, sympy.Integer):
            out = _Integer(expr, make_array)
        elif isinstance(expr, sympy.Float):
            out = _Float(expr, make_array)
        elif isinstance(expr, sympy.Rational):
            out = _Rational(expr, make_array)
        elif expr in (sympy.E, sympy.pi, sympy.EulerGamma, sympy.I):
            out = _Constant(expr, make_array)
        else:
            out = _Func(expr, memodict, func_lookup, make_array)
        memodict[expr] = out
        return out


def _is_node(x):
    return isinstance(x, _AbstractNode)


class SymbolicModule(eqx.Module):
    nodes: PyTree
    has_extra_funcs: bool = eqx.static_field()

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
        self.nodes = jax.tree_map(_convert, expressions)

    def sympy(self) -> sympy.Expr:
        if self.has_extra_funcs:
            raise NotImplementedError(
                "SymbolicModule cannot be converted back to SymPy if `extra_funcs` "
                "is passed."
            )
        memodict = dict()
        return jax.tree_map(
            lambda n: n.sympy(memodict, _reverse_lookup), self.nodes, is_leaf=_is_node
        )

    def __call__(self, **symbols):
        memodict = symbols
        return jax.tree_map(lambda n: n(memodict), self.nodes, is_leaf=_is_node)
