import logging
import re

import clang.cindex
import clang.enumerations

logger = logging.getLogger(__name__)

try:
    clang.cindex.Config.set_library_path(
        "/work/LAS/weile-lab/benjis/weile-lab/thesis/ReVeal/clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04/lib")
    clang.cindex.Config.set_library_file(
        '/work/LAS/weile-lab/benjis/weile-lab/thesis/ReVeal/clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04/lib/libclang.so.6.0')
except Exception:
    logger.error(f'could not set up clang!')


class Tokenizer:
    # creates the object, does the inital parse
    def __init__(self, path, tokenizer_type='original'):
        self.index = clang.cindex.Index.create()
        self.tu = self.index.parse(path)
        self.path = self.extract_path(path)
        self.symbol_table = {}
        self.symbol_count = 1
        self.tokenizer_type = tokenizer_type

    # To output for split_functions, must have same path up to last two folders
    def extract_path(self, path):
        return "".join(path.split("/")[:-2])

    def full_tokenize_cursor(self, cursor):
        tokens = cursor.get_tokens()
        result = []
        for token in tokens:
            if token.kind.name == "COMMENT":
                continue
            if token.kind.name == "LITERAL":
                result += self.process_literal(token)
                continue
            if token.kind.name == "IDENTIFIER":
                result += ["ID"]
                continue
            result += [token.spelling]
        return result

    def full_tokenize(self):
        cursor = self.tu.cursor
        return self.full_tokenize_cursor(cursor)

    def process_literal(self, literal):
        cursor_kind = clang.cindex.CursorKind
        kind = literal.cursor.kind
        if kind == cursor_kind.INTEGER_LITERAL:
            return literal.spelling
        if kind == cursor_kind.FLOATING_LITERAL:
            return literal.spelling
        if kind == cursor_kind.IMAGINARY_LITERAL:
            return ["NUM"]
        if kind == cursor_kind.STRING_LITERAL:
            return ["STRING"]
        sp = literal.spelling
        if re.match('[0-9]+', sp) is not None:
            return sp
        return ["LITERAL"]

    def split_functions(self, method_only):
        results = []
        cursor_kind = clang.cindex.CursorKind
        cursor = self.tu.cursor
        for c in cursor.get_children():
            filename = c.location.file.name if c.location.file != None else "NONE"
            extracted_path = self.extract_path(filename)

            if (c.kind == cursor_kind.CXX_METHOD or (
                    method_only == False and c.kind == cursor_kind.FUNCTION_DECL)) and extracted_path == self.path:
                name = c.spelling
                tokens = self.full_tokenize_cursor(c)
                filename = filename.split("/")[-1]
                results += [tokens]

        return results


def tokenize(file_text):
    try:
        c_file = open('/tmp/test1.c', 'w')
        c_file.write(file_text)
        c_file.close()
        tok = Tokenizer('/tmp/test1.c')
        results = tok.split_functions(False)
        return ' '.join(results[0])
    except:
        return None
