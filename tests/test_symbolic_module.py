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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import sympy

import sympy2jax


def assert_equal(x, y):
    x_leaves, x_tree = jtu.tree_flatten(x)
    y_leaves, y_tree = jtu.tree_flatten(y)
    assert x_tree == y_tree
    for xi, yi in zip(x_leaves, y_leaves):
        assert type(xi) is type(yi)
        if isinstance(xi, jnp.ndarray):
            assert xi.shape == yi.shape
            assert xi.dtype == yi.dtype
            assert jnp.all(xi == yi)
        else:
            assert xi == yi


def assert_sympy_allclose(x, y):
    assert isinstance(x, sympy.Expr)
    assert isinstance(y, sympy.Expr)
    assert x.func is y.func
    if isinstance(x, sympy.Float):
        assert abs(float(x) - float(y)) < 1e-5
    elif isinstance(x, sympy.Integer):
        assert x == y
    elif isinstance(x, sympy.Rational):
        assert x.numerator == y.numerator  # pyright: ignore
        assert x.denominator == y.denominator  # pyright: ignore
    elif isinstance(x, sympy.Symbol):
        assert x.name == y.name  # pyright: ignore
    else:
        assert len(x.args) == len(y.args)
        for xarg, yarg in zip(x.args, y.args):
            assert_sympy_allclose(xarg, yarg)


def test_example():
    x_sym = sympy.symbols("x_sym")
    cosx = 1.0 * sympy.cos(x_sym)  # pyright: ignore[reportOperatorIssue]
    sinx = 2.0 * sympy.sin(x_sym)  # pyright: ignore[reportOperatorIssue]
    mod = sympy2jax.SymbolicModule([cosx, sinx])

    x = jax.numpy.zeros(3)
    out = mod(x_sym=x)
    params = jtu.tree_leaves(mod)

    assert_equal(out, [jnp.cos(x), 2 * jnp.sin(x)])
    assert_equal(
        [x for x in params if eqx.is_array(x)], [jnp.array(1.0), jnp.array(2.0)]
    )


def test_grad():
    x_sym = sympy.symbols("x_sym")
    y = 2.1 * x_sym**2
    mod = sympy2jax.SymbolicModule(y)
    x = jnp.array(1.1)

    grad_m = eqx.filter_grad(lambda m, z: m(x_sym=z))(mod, x)
    grad_z = eqx.filter_grad(lambda z, m: m(x_sym=z))(x, mod)

    true_grad_m = eqx.filter(
        sympy2jax.SymbolicModule(1.21 * x_sym**2), eqx.is_inexact_array
    )
    true_grad_z = jnp.array(4.2 * x)

    assert_equal(grad_m, true_grad_m)
    assert_equal(grad_z, true_grad_z)

    mod2 = eqx.apply_updates(mod, grad_m)
    expr = mod2.sympy()

    assert_sympy_allclose(expr, 3.31 * x_sym**2)


def test_reduce():
    x, y, z = sympy.symbols("x y z")
    z = 2 * x * y * z
    mod = sympy2jax.SymbolicModule(expressions=z)
    mod(x=jnp.array(0.4), y=jnp.array(0.5), z=jnp.array(0.6))

    z = 2 + x + y + z
    mod = sympy2jax.SymbolicModule(expressions=z)
    mod(x=jnp.array(0.4), y=jnp.array(0.5), z=jnp.array(0.6))


def test_special_subclasses():
    x, y = sympy.symbols("x y")
    z = x - 1  # sympy.core.numbers.NegativeOne
    w = y * 0  # sympy.core.numbers.Zero
    v = x + 1 / 2  # sympy.core.numbers.OneHalf

    mod = sympy2jax.SymbolicModule([z, w, v])
    assert_equal(mod(x=1, y=1), [jnp.array(0), jnp.array(0), jnp.array(1.5)])
    assert mod.sympy() == [z, sympy.Integer(0), v]


def test_rational():
    x = sympy.symbols("x")
    y = x + sympy.Integer(3) / sympy.Integer(7)
    mod = sympy2jax.SymbolicModule(y)
    assert mod(x=1.0) == 1 + 3 / 7
    assert mod.sympy() == y


def test_constants():
    x = sympy.symbols("x")
    y = x + sympy.pi + sympy.E + sympy.EulerGamma + sympy.I
    mod = sympy2jax.SymbolicModule(y)
    assert jnp.isclose(mod(x=1.0), 1 + jnp.pi + jnp.e + jnp.euler_gamma + 1j)
    assert mod.sympy() == y


def test_extra_funcs():
    class _MLP(eqx.Module):
        mlp: eqx.nn.MLP

        def __init__(self):
            self.mlp = eqx.nn.MLP(1, 1, 2, 2, key=jr.PRNGKey(0))

        def __call__(self, x):
            x = jnp.asarray(x)
            return self.mlp(x[None])[0]

    expr = sympy.parsing.sympy_parser.parse_expr("f(x) + y")
    mlp = _MLP()
    mod = sympy2jax.SymbolicModule(expr, {sympy.Function("f"): mlp})
    mod(x=1.0, y=2.0)

    def _get_params(module):
        return {id(x) for x in jtu.tree_leaves(module) if eqx.is_array(x)}

    assert _get_params(mod).issuperset(_get_params(mlp))


def test_concatenate():
    x, y, z = sympy.symbols("x y z")
    cat = sympy2jax.concatenate(x, y, z)
    mod = sympy2jax.SymbolicModule(expressions=cat)
    assert_equal(
        mod(x=jnp.array([0.4, 0.5]), y=jnp.array([0.6, 0.7]), z=jnp.array([0.8, 0.9])),
        jnp.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    )


def test_stack():
    x, y, z = sympy.symbols("x y z")
    stack = sympy2jax.stack(x, y, z)
    mod = sympy2jax.SymbolicModule(expressions=stack)
    assert_equal(
        mod(x=jnp.array(0.4), y=jnp.array(0.5), z=jnp.array(0.6)),
        jnp.array([0.4, 0.5, 0.6]),
    )


def test_non_array_to_sympy():
    mod = sympy2jax.SymbolicModule(expressions=[sympy.Integer(1)], make_array=False)
    assert mod.sympy() == [sympy.Integer(1)]
