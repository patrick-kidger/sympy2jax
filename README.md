<h1 align="center">sympy2jax</h1>

Turn SymPy expressions into trainable JAX expressions. The output will be an [Equinox](https://github.com/patrick-kidger/equinox) module with all SymPy floats (integers, rationals, ...) as leaves. SymPy symbols will be inputs.

Optimise your symbolic expressions via gradient descent!

## Installation

```bash
pip install sympy2jax
```

Requires:  
Python 3.7+  
JAX 0.3.4+  
Equinox 0.5.3+  
SymPy 1.7.1+.

## Example

```python
import jax
import sympy
import sympy2jax

x_sym = sympy.symbols("x_sym")
cosx = 1.0 * sympy.cos(x_sym)
sinx = 2.0 * sympy.sin(x_sym)
mod = sympy2jax.SymbolicModule([cosx, sinx])  # PyTree of input expressions

x = jax.numpy.zeros(3)
out = mod(x_sym=x)  # PyTree of results.
params = jax.tree_leaves(mod)  # 1.0 and 2.0 are parameters.
                               # (Which may be trained in the usual way for Equinox.)
```

## Documentation

```python
sympytorch.SymbolicModule(expressions, extra_funcs=None, make_array=True)
```

Where:
- `expressions` is a PyTree of SymPy expressions.
- `extra_funcs` is an optional dictionary from SymPy functions to JAX operations, to extend the built-in translation rules.
- `make_array` is whether integers/floats/rationals should be stored as Python integers/etc., or as JAX arrays.

Instances can be called with key-value pairs of symbol-value, as in the above example.

Instances have a `.sympy()` method that translates the module back into a PyTree of SymPy expressions.

(That's literally the entire documentation, it's super easy.)

## Finally

### See also: other tools in the JAX ecosystem

Neural networks: [Equinox](https://github.com/patrick-kidger/equinox).

Numerical differential equation solvers: [Diffrax](https://github.com/patrick-kidger/diffrax).

Type annotations and runtime checking for PyTrees and shape/dtype of JAX arrays: [jaxtyping](https://github.com/google/jaxtyping). 

### Disclaimer

This is not an official Google product.
