# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

### Getting started

[We assume that you have `uv` installed.](https://docs.astral.sh/uv/) Now fork the library on GitHub. Then clone and install the library:

```bash
git clone https://github.com/your-username-here/sympy2jax.git
cd sympy2jax
uv run prek install  # Creates a local venv + installs dependencies + installs pre-commit hooks.
```

---

### If you're making changes to the code

Now make your changes. Make sure to include additional tests if necessary. Next verify the tests all pass:

```bash
uv run pytest
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!
