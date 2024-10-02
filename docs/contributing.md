## The Algorithm

Coppafish is built on the basis that an algorithm that performs well does not need to be changed. The algorithm is only 
updated when there is evidence that it can perform better and that the current algorithm is performing worse.

Typically there is a protected staging branch, like `v1.0.0`, for a future release. This must be pull requested into 
and must pass integration tests. The `main` branch is kept at the latest version release.

## Code Philosophy

We follow basic rules when coding. Anyone can code something that works, but coding it in a scaleable, maintainable way 
is another struggle altogether.

Here are some specific standards to follow:

* Knowledge written down twice is bad code. Don't Repeat Yourself (DRY)!
* If a bug is found, the bug must be automatically found if it is to occur again.
* Every time a function is modified or created, a new unit test must be created for the function. A pre-existing unit 
test can be drawn from to build a new unit test, but it should be clear in your mind that you are affectively building 
a new function.
* Minimise `#!python if`/`#!python else` branching as much as possible. Exit `#!python if`/`#!python else` nesting as 
soon as possible through the use of keywords like `#!python continue`, `#!python break` and `#!python return`, whenever 
feasible.
* Do not over-shorten a variable or function name.
* In most cases, a line of code should do only one operation.
* Every docstring for a function must be complete so a developer can re-create the function without seeing any of the 
existing source code.
* Each parameter in a function must have an independent, clear functionality. If two parameters are derivable from 
one another, you are doing something wrong. This also applies for the function's return variables.
* Minimise the number of data types a parameter can be and use common sense. For example, a parameter that can be 
`#!python int` or `#!python None` is reasonable. A parameter that can be `#!python bool` or `#!python float` is not 
reasonable.
* The documentation should update in parallel with the code. Having the documentation as part of the github repository 
makes this easier.

## Run Tests

In your coppafish environment, install dev packages 

```terminal
pip install -r requirements-dev.txt
```

Run unit tests (~10s) 

```terminal
pytest -m "not integration and not manual and not notebook"
```

Run integration tests (~50s) 

```terminal
pytest -m "integration and not manual"
```

Run unit tests requiring a notebook (~3s) 

```terminal
pytest -m "notebook and not integration and not manual"
```

## Run Documentation Locally

Install [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) 

```terminal
python -m pip install mkdocs-material
```

Start the documentation locally 

```terminal
mkdocs serve
```
