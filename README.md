Density Functional Theorem for Dummies
======================================

DFT dummy is a recreational project that is built to remind me of the olden days,
when I was doing some material simulations using all kinds of quantum software.
There are certain problems that I am not 100% sure about and would really like
to spend some time learning, meanwhile sharpening my software skills.

Python will be the main programming language of this project because it is easy
to use, tons of good numerical software and let me focus on the physics and math
problems without thinking about how memory is allocated, etc.

My intention with the project is to write down as many thoughts as possible with
minimal amount of code. It will be very common to see me put down tens of lines of
comments for a single line of code.

I will organise the code as I go and I don't expect to make this a full-fledged
quantum computing toolkit. The project will remain educational until I think it is
production ready.

I will borrow thoughts from other mature projects and write them down in my own
understanding. Certainly, there will be some similarity and references will be given.

## Install with Poetry

```
export SYSTEM_VERSION_COMPAT=1 # for macos
poetry install --no-root # for dev
```

## Test

```
pytest
```