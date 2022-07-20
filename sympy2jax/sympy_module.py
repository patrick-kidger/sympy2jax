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
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import sympy


PyTree = Any


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)

    return fn_


_lookup = {
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

_reverse_lookup = {v: k for k, v in _lookup.items()}
assert len(_reverse_lookup) == len(_lookup)


class _IdDict:
    def __init__(self, **values):
        self._dict = {id(k): v for k, v in values.items()}

    def __getitem__(self, item):
        return self._dict[id(item)]

    def __setitem__(self, item, value):
        self._dict[id(item)] = value


class _AbstractNode(eqx.Module):
    @abc.abstractmethod
    def __call__(self, memodict: _IdDict):
        ...

    @abc.abstractmethod
    def sympy(self, memodict: _IdDict, func_lookup: dict) -> sympy.Expr:
        ...


class _Symbol(_AbstractNode):
    _name: str

    def __init__(self, expr: sympy.Expr):
        self._name = expr.name

    def __call__(self, memodict: _IdDict):
        try:
            return memodict[self._name]
        except KeyError as e:
            raise KeyError(f"Missing input for symbol {self._name}") from e

    def sympy(self, memodict: _IdDict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Symbol(self._name)


class _Integer(_AbstractNode):
    _value: jnp.ndarray

    def __init__(self, expr: sympy.Expr):
        assert isinstance(expr, sympy.Integer)
        self._value = jnp.asarray(int(expr))

    def __call__(self, memodict: _IdDict):
        return self._value

    def sympy(self, memodict: _IdDict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Integer(self._value.item())


class _Float(_AbstractNode):
    _value: jnp.ndarray

    def __init__(self, expr: sympy.Expr):
        assert isinstance(expr, sympy.Float)
        self._value = jnp.asarray(float(expr))

    def __call__(self, memodict: _IdDict):
        return self._value

    def sympy(self, memodict: _IdDict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Float(self._value.item())


class _Rational(_AbstractNode):
    _numerator: jnp.ndarray
    _denominator: jnp.ndarray

    def __init__(self, expr: sympy.Expr):
        assert isinstance(expr, sympy.Rational)
        self._numerator = jnp.asarray(int(expr.numerator))
        self._denominator = jnp.asarray(int(expr.denominator))

    def __call__(self, memodict: _IdDict):
        return self._numerator / self._denominator

    def sympy(self, memodict: _IdDict, func_lookup: dict) -> sympy.Expr:
        # memodict not needed as sympy deduplicates internally
        return sympy.Integer(self._numerator) / sympy.Integer(self._denominator)


class _Func(_AbstractNode):
    _func: Callable
    _args: list

    def __init__(self, expr: sympy.Expr, memodict: _IdDict, func_lookup: dict):
        try:
            self._func = func_lookup[expr.func]
        except KeyError as e:
            raise KeyError(f"Unsupported Sympy type {type(expr)}") from e
        self._args = [_sympy_to_node(arg, memodict, func_lookup) for arg in expr.args]

    def __call__(self, memodict: _IdDict):
        args = []
        for arg in self._args:
            try:
                arg_call = memodict[arg]
            except KeyError:
                arg_call = arg(memodict)
                memodict[arg] = arg_call
            args.append(arg_call)
        return self._func(*args)

    def sympy(self, memodict: _IdDict, func_lookup: dict) -> sympy.Expr:
        try:
            return memodict[self]
        except KeyError:
            func = func_lookup[self._func]
            args = [arg.sympy(memodict, func_lookup) for arg in self._args]
            out = func(*args)
            memodict[self] = out
            return out


def _sympy_to_node(
    expr: sympy.Expr, memodict: _IdDict, func_lookup: dict
) -> _AbstractNode:
    try:
        return memodict[expr]
    except KeyError:
        if isinstance(expr, sympy.Symbol):
            out = _Symbol(expr)
        elif isinstance(expr, sympy.Integer):
            out = _Integer(expr)
        elif isinstance(expr, sympy.Float):
            out = _Float(expr)
        elif isinstance(expr, sympy.Rational):
            out = _Rational(expr)
        else:
            out = _Func(expr, memodict, func_lookup)
        memodict[expr] = out
        return out


def _is_node(x):
    return isinstance(x, _AbstractNode)


class SymbolicModule(eqx.Module):
    nodes: PyTree
    has_extra_funcs: bool = eqx.static_field()

    def __init__(
        self, expressions: PyTree, extra_funcs: Optional[dict] = None, **kwargs
    ):
        super().__init__(**kwargs)
        if extra_funcs is None:
            lookup = _lookup
            self.has_extra_funcs = False
        else:
            lookup = co.ChainMap(extra_funcs, _lookup)
            self.has_extra_funcs = True
        _convert = ft.partial(_sympy_to_node, memodict=_IdDict(), func_lookup=lookup)
        self.nodes = jax.tree_map(_convert, expressions)

    def sympy(self) -> sympy.Expr:
        if self.has_extra_funcs:
            raise NotImplementedError(
                "SymbolicModule cannot be converted back to SymPy if `extra_funcs` is passed"
            )
        memodict = _IdDict()
        return jax.tree_map(
            lambda n: n.sympy(memodict, _reverse_lookup), self.nodes, is_leaf=_is_node
        )

    def __call__(self, **symbols):
        memodict = _IdDict(**symbols)
        return jax.tree_map(lambda n: n(memodict), self.nodes, is_leaf=_is_node)
