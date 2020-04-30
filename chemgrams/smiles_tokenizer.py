import collections
import re


SMILESToken = collections.namedtuple('SMILESToken', ['type', 'value', 'index'])


class SMILESTokenizer:
    """
    A SMILES token consists of an atom (e.g. "C", or "[O+]"), a ring size specifier (e.g. "1", or "%10"),
    a grouping symbol (i.e. "(" and ")"), or a bond (e.g. "=").

    Atoms:
    [H]
    B, b, [B-], [B], [BH-], [BH], [B@-], [B@@-], [BH3-]
    C, c, [C@@H], [C@H], [C@], [C@@], [CH2-], [CH-], [C], [C+], [C-], [c], [c-], [CH+], [CH], [CH2+], [CH2], [C@H-]
    O, o, [o+], [O-], [OH+], [O+], [OH2], [O], [OH]
    N, n, [NH+], [nH], [NH3+], [NH2+], [N-], [n+], [NH-], [N+], [nH+], [n-], [NH4+], [N], [N@@+], [N@+], [N@@H+], [N@H+]
    S, s, [S@], [S@@], [S-], [S+],[SH+], [s+], [S@@+], [S@+], [S], [SH-], [s@@], [S@H], [S@@H]
    P, p, [P@@H], [P@@], [PH2], [P@], [P@H], [P@H+], [P+], [PH+], [PH], [P@@H+], [P], [P@@+], [PH2+], [P@+]
    F, Cl, Br, I, [F], [Cl], [Br], [I]

    Ring Sizes:
    1, 2, 3, 4, 5, 6, 7, 8, 9, %10, %(122), etc.

    Bonds:
    /, \, =, #, -, :, .

    Branches:
    ()
    """
    def __init__(self, smiles):
        self._smiles = smiles
        self._tokens = self._tokenize()
        self._validate()

    def _tokenize(self):

        ring_closure = "[1-9]|%[1-9][0-9]|%\([1-9][0-9]{2,}\)"
        ring_closure_group = "(?P<ring_closure>%s)" % ring_closure

        formal_atoms = ("[H]", "[B-]", "[B]", "[C@@H]", "[C@H]", "[C@]", "[C@@]", "[CH2-]", "[CH-]", "[C]", "[C+]", "[C-]", "[c-]", "[C@H-]",
                        "[o+]", "[O-]", "[OH+]", "[O+]", "[OH2]", "[O]", "[N@@+]", "[N@+]", "[N@@H+]", "[c]", "[BH-]", "[CH+]", "[CH]",
                        "[NH+]", "[nH]", "[NH3+]", "[NH2+]", "[N-]", "[n+]", "[NH-]", "[N+]", "[nH+]", "[n-]", "[NH4+]", "[N]", "[N@H+]",
                        "[S@]", "[S@@]", "[S-]", "[S+]", "[SH+]", "[s+]", "[S@@+]", "[S@+]", "[S]", "[SH-]", "[s@@]", "[OH]", "[CH2]",
                        "[P@@]", "[P@@H]", "[P@@H+]", "[P@]", "[P@H]", "[P@H+]", "[P+]", "[PH]", "[PH+]", "[PH2]", "[P]", "[PH2+]",
                        "[F]", "[Cl]", "[Br]", "[I]", "[P@@+]", "[P@+]", "[BH]", "[B@-]", "[B@@-]", "[BH3-]", "[CH2+]", "[S@H]", "[S@@H]")
        escaped = ()
        for atom in formal_atoms:
            escaped += (re.escape(atom),)
        formal_atom_group = "(?P<formal_atom>%s)" % '|'.join(escaped)

        plain_atom_group = "(?P<plain_atom>Cl|Br|B|b|C|c|O|o|N|n|S|s|P|p|F|I)"

        bonds = ("=", "\\", "/", "-", "#", ":", ".")
        escaped = ()
        for bond in bonds:
            escaped += (re.escape(bond),)
        bond_group = "(?P<bond>%s)" % '|'.join(escaped)

        branch_group = "(?P<grouping>\(|\))"
        regex = "%s|%s|%s|%s|%s" % (formal_atom_group, plain_atom_group, ring_closure_group, bond_group, branch_group)

        tokens = []
        for mo in re.finditer(regex, self._smiles):
            kind = mo.lastgroup
            value = mo.group()
            index = mo.start()
            tokens.append(SMILESToken(kind, value, index))
        return tokens

    def _validate(self):
        token_str = ""
        for token in self._tokens:
            token_str += token.value
        if token_str != self._smiles:
            raise Exception("could not tokenize SMILES string: %s (token string: %s)" % (self._smiles, token_str))

    def get_tokens(self):
        return self._tokens

    def get_raw_length(self):
        """
        Returns the number of characters in the string.
        :return: an integer representing the number of characters in the string.
        """
        return len(self._smiles)

    def get_tokenized_length(self):
        """
        Returns the number of SMILES tokens in the string.
        :return: an integer representing the number of SMILES tokens in the string.
        """
        return len(self._tokens)

    def token_at(self, tokenized_index):
        """
        Returns the SMILES token at the tokenized index.
        :param index:
        :return:
        """
        return self._tokens[tokenized_index]
