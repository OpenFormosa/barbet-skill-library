"""Internal subprocess entry point. Never import from an inference model."""
import ast
import builtins
import collections
import contextlib
import ctypes
import errno
import fractions
import functools
import heapq
import io
import itertools
import json
import math
import operator
import os
import random
import re
import resource
import statistics
import sys

MODULES = {m.__name__: m for m in (math, re, collections, itertools, functools,
                                 heapq, fractions, statistics, operator, random)}


def guard_tree(code):
    tree = ast.parse(code)
    forbidden = {'eval', 'exec', 'compile', 'open', 'input', 'getattr', 'setattr',
                 'delattr', 'globals', 'locals', 'vars', 'dir', 'breakpoint',
                 '__import__', '__builtins__', 'help'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module]
            if any(name not in MODULES for name in names):
                raise ValueError('unsupported import: ' + str(names))
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError('relative imports not supported')
            if any(x.name.startswith('_') or (x.asname or '').startswith('_') for x in node.names):
                raise ValueError('private import')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('private attribute access')
        if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith('__') and node.id != '__name__'):
            raise ValueError('unsupported name: ' + node.id)
    return tree


def isolate():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not ctx:
        raise RuntimeError('seccomp initialization failed')
    allowed = ('read write close fstat newfstatat lseek mmap mprotect munmap brk '
               'rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack futex '
               'clock_gettime gettimeofday getrusage getpid gettid '
               'getrandom sched_yield madvise mremap exit exit_group').split()
    for name in allowed:
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number >= 0 and lib.seccomp_rule_add(ctx, 0x7fff0000, number, 0) != 0:
            raise RuntimeError('seccomp rule failed: ' + name)
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    if lib.seccomp_load(ctx) != 0:
        raise RuntimeError('seccomp load failed')
    lib.seccomp_release(ctx)


def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in MODULES:
        raise ImportError('unsupported module: ' + name)
    return MODULES[name]


class BoundedOutput(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 8192:
            raise ValueError('stdout limit exceeded')
        return super().write(text)


def main():
    request = json.loads(sys.stdin.read(220_000))
    try:
        code = compile(guard_tree(request['code']), '<model>', 'exec')
        setup = compile(guard_tree(request['setup']), '<public_setup>', 'exec')
        tests = [compile(guard_tree(x), '<check>', 'exec') for x in request['tests']]
        # Preload encodings/formatting before filesystem access is disabled.
        json.dumps({'probe': '繁體中文'}, ensure_ascii=False)
        isolate()
    except BaseException as exc:
        print(json.dumps({'passed': False, 'status': 'isolation_or_compile_failed',
                          'error': f'{type(exc).__name__}: {str(exc)[:1000]}'}))
        return
    names = ('abs all any bin bool bytearray bytes callable chr complex dict divmod '
             'enumerate filter float format frozenset hash hex int isinstance issubclass '
             'iter len list map max min next oct ord pow print range repr reversed round '
             'set slice sorted str sum tuple type zip object super property staticmethod '
             'classmethod Exception ValueError TypeError IndexError KeyError StopIteration '
             'AssertionError ZeroDivisionError').split()
    safe = {name: getattr(builtins, name) for name in names}
    safe['__import__'] = limited_import
    safe['__build_class__'] = builtins.__build_class__
    namespace = {'__builtins__': safe, '__name__': '__candidate__'}
    output = BoundedOutput()
    checks_run = 0
    result = {'passed': False, 'status': 'execution_failed', 'answer': None}
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(setup, namespace)
            exec(code, namespace)
            for test in tests:
                exec(test, namespace)
                checks_run += 1
        answer = namespace.get('answer') if request['want_answer'] else None
        if request['want_answer']:
            if isinstance(answer, fractions.Fraction):
                answer = str(answer)
            elif type(answer) not in (int, float, str) or isinstance(answer, float) and not math.isfinite(answer):
                raise ValueError('answer must be a finite number or numeric string')
        result.update(passed=True, status='executed_checks_passed', answer=answer)
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {str(exc)[:1000]}'
    result.update(stdout=output.getvalue()[-2000:], checks_run=checks_run)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
