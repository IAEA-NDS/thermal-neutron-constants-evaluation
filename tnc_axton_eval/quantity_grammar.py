from lark import Lark, Transformer, v_args
from quantities import prepare_funcs


calc_grammar = """
// Adopted from the calculator example at
// https://lark-parser.readthedocs.io/en/stable/examples/calc.html 
// but without the fancy tree shaping directives explained at 
// https://lark-parser.readthedocs.io/en/stable/tree_construction.html

?start: sum

?sum: product
    | sum "+" product   -> add
    | sum "-" product   -> sub

?product: atom
    | product "*" atom  -> mul
    | product "/" atom  -> div

?atom: NUMBER           -> number
     | "-" atom         -> neg
     | func
     | "(" sum ")"

func: NAME ("(" NUMBER ("," NUMBER)* ")")?

%import common.CNAME -> NAME
%import common.NUMBER -> NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""


@v_args(inline=True)    # Affects the signatures of the methods
class CalculateTree(Transformer):
    from operator import add, sub, mul, truediv as div, neg
    number = float

    def __init__(self, funcs):
        self._funcs = funcs

    def func(self, funcname, *args):
        if not self._funcs.is_valid(funcname):
            raise ValueError(f'invalid function name {funcname}')
        ret = getattr(self._funcs, funcname)(*[int(arg) for arg in args])
        return ret


def prepare_propagator(funcs):
    trafo = CalculateTree(funcs)
    parser = Lark(calc_grammar, parser='lalr', transformer=trafo)
    return parser.parse
