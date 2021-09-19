import math
import random
import re
from functools import reduce

import numpy as np
from termcolor import colored
from fractions import Fraction

# Supports multiplications, division, powers, +/-


class ComplexNumber:
    # z = a + bi
    def __init__(self, real=Fraction(0), imaginary=Fraction(0)):
        self.real: Fraction = real
        if not isinstance(self.real, Fraction):
            if isinstance(self.real, float):
                self.real = Fraction(str(real))
            else:
                self.real = Fraction(self.real)
        self.imaginary: Fraction = imaginary
        if not isinstance(self.imaginary, Fraction):
            if isinstance(self.imaginary, float):
                self.imaginary = Fraction(str(imaginary))
            else:
                self.imaginary = Fraction(self.imaginary)

    def get_common_denominator(self):
        return lcm(self.real.denominator, self.imaginary.denominator)

    def get_numerator_gcd(self):
        return math.gcd(abs(self.real.numerator), abs(self.imaginary.numerator))

    def __add__(self, o):
        if isinstance(o, (int, float, Fraction)):
            return self + ComplexNumber(real=o, imaginary=Fraction(0))
        elif isinstance(o, ComplexNumber):
            return ComplexNumber(self.real + o.real, self.imaginary + o.imaginary)
        else:
            raise NotImplementedError

    def __radd__(self, o):
        return self.__add__(o)

    def __neg__(self):
        return ComplexNumber(-self.real, -self.imaginary)

    def __pos__(self):
        return self

    def __sub__(self, o):
        if isinstance(o, (int, float, Fraction)):
            return self - ComplexNumber(real=o, imaginary=Fraction(0))
        elif isinstance(o, ComplexNumber):
            return ComplexNumber(self.real - o.real, self.imaginary - o.imaginary)
        else:
            raise NotImplementedError

    def __rsub__(self, o):
        return -1 * (self - o)

    def __mul__(self, o):
        if isinstance(o, (int, float, Fraction)):
            return ComplexNumber(real=self.real * o, imaginary=self.imaginary * o)
        elif isinstance(o, ComplexNumber):
            return ComplexNumber(self.real * o.real - self.imaginary * o.imaginary, self.real * o.imaginary + self.imaginary * o.real)
        else:
            raise NotImplementedError

    def __pow__(self, power):
        if isinstance(power, int):
            if power == 0:
                return ComplexNumber(1, 0)
            elif power > 0:
                z = self
                for _ in range(power - 1):
                    z = self * z
                return z
            elif power < 0:
                if self.real ** 2 + self.imaginary ** 2 == 0:
                    raise ZeroDivisionError
                inv = 1 / self
                return inv ** (-power)
        elif isinstance(power, float):
            if not power.is_integer():
                raise NotImplementedError
            return self ** int(power)
        elif isinstance(power, Fraction):
            if power.denominator != 1:
                raise NotImplementedError
            return self ** int(power.numerator)
        else:
            raise NotImplementedError

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        if isinstance(o, (int, float, Fraction)):
            if o == 0:
                raise ZeroDivisionError
            return ComplexNumber(real=self.real / o, imaginary=self.imaginary / o)
        elif isinstance(o, ComplexNumber):
            if o.real ** 2 + o.imaginary ** 2 == 0:
                raise ZeroDivisionError
            inv = ComplexNumber(o.real / (o.real ** 2 + o.imaginary ** 2), -o.imaginary / (o.real ** 2 + o.imaginary ** 2))
            return self * inv
        else:
            raise NotImplementedError

    def __rtruediv__(self, o):
        if isinstance(o, (int, float, Fraction)):
            return ComplexNumber(o).__truediv__(self)
        elif isinstance(o, ComplexNumber):
            return o.__truediv__(o)
        else:
            raise NotImplementedError

    def __str__(self):
        if self.real == self.imaginary == 0:
            return "0"
        elif self.real == 0 and self.imaginary != 0:
            return (str(self.imaginary) + 'i').replace('1i', 'i')
        elif self.real != 0 and self.imaginary == 0:
            return str(self.real)

        if self.imaginary > 0:
            return (str(self.real) + " + " + str(self.imaginary) + "i").replace('1i', 'i')
        else:
            return (str(self.real) + " - " + str(self.imaginary * -1) + "i").replace('1i', 'i')

    @staticmethod
    def is_frac(val: Fraction):
        return '/' in str(val)

    def process_tex(self) -> str:
        if self.real == self.imaginary == 0:
            return "0"
        elif self.real == 0 and self.imaginary != 0:
            if self.is_frac(self.imaginary):
                return ('-' * (self.imaginary < 0) + '\\frac{' + str(abs(int(self.imaginary.numerator))) + 'i}{' + str(abs(int(self.imaginary.denominator))) + '}').replace('1i', 'i')
            else:
                return (str(self.imaginary) + 'i').replace('1i', 'i')
        elif self.real != 0 and self.imaginary == 0:
            if self.is_frac(self.real):
                return '-' * (self.real < 0) + '\\frac{' + str(abs(int(self.real.numerator))) + '}{' + str(abs(int(self.real.denominator))) + '}'
            else:
                return str(self.real)
        else:
            re = abs(self.real)
            im = abs(self.imaginary)

            if self.is_frac(im):
                im_tex = ('\\frac{' + str(abs(int(im.numerator))) + 'i}{' + str(abs(int(im.denominator))) + '}').replace('{1i', '{i')
            else:
                if im == 1:
                    im_tex = 'i'
                else:
                    im_tex = (str(im) + 'i').replace('1i', 'i')

            if self.is_frac(re):
                re_tex = '\\frac{' + str(abs(int(re.numerator))) + '}{' + str(abs(int(re.denominator))) + '}'
            else:
                re_tex = str(re)

            if self.real > 0 and self.imaginary > 0:
                return '(' + re_tex + ' + ' + im_tex + ')'
            elif self.real > 0 and self.imaginary < 0:
                return '(' + re_tex + ' - ' + im_tex + ')'
            elif self.real < 0 and self.imaginary > 0:
                return '-(' + re_tex + ' - ' + im_tex + ')'
            else:
                return '-(' + re_tex + ' + ' + im_tex + ')'

    def __eq__(self, o):
        if isinstance(o, ComplexNumber):
            return self.real == o.real and self.imaginary == o.imaginary
        elif isinstance(o, (int, float, Fraction)):
            return self.imaginary == 0 and self.real == o
        else:
            raise NotImplementedError


class Polynomial:
    def __init__(self, coefficients: list, variable='x'):
        self.variable = variable
        self.deg = len(coefficients) - 1
        self.coefficients = coefficients[:]
        self.coefficients = [item if isinstance(item, ComplexNumber) else ComplexNumber(item) for item in self.coefficients]
        while self.coefficients[0] == 0 and self.deg > 0:
            self.deg -= 1
            self.coefficients = self.coefficients[1:]

    def copy(self):
        return Polynomial(self.coefficients[:])

    def is_number(self):
        return len(self.coefficients) == 1

    def __add__(self, o):
        if isinstance(o, Polynomial):
            if self.deg >= o.deg:
                large = self.coefficients[:]
                small = o.coefficients[:]
            else:
                small = self.coefficients[:]
                large = o.coefficients[:]
            for i in range(len(small)):
                large[-i] += small[-i]
            return Polynomial(large)
        elif isinstance(o, (int, float, Fraction, ComplexNumber)):
            return Polynomial([o]) + self
        else:
            raise NotImplementedError

    def __radd__(self, o):
        if isinstance(o, (int, float, Fraction, ComplexNumber)):
            return self + o
        else:
            raise NotImplementedError

    def __sub__(self, o):
        if isinstance(o, Polynomial):
            return self + Polynomial(list(map(lambda x: -x, o.coefficients)))
        elif isinstance(o, (int, float, Fraction, ComplexNumber)):
            return self - Polynomial([o])
        else:
            raise NotImplementedError

    def __rsub__(self, o):
        if isinstance(o, (int, float, Fraction, ComplexNumber)):
            return Polynomial(o) - self
        else:
            raise NotImplementedError

    def __mul__(self, o):
        if isinstance(o, (int, float, Fraction, ComplexNumber)):
            coef = self.coefficients[:]
            coef = list(map(lambda x: x * o, coef))
            return Polynomial(coef)
        elif isinstance(o, Polynomial):
            new_coef = [0] * (len(self.coefficients) + len(o.coefficients) - 1)
            for p1, coef1 in enumerate(self.coefficients[::-1]):
                for p2, coef2 in enumerate(o.coefficients[::-1]):
                    new_coef[p1+p2] += coef1 * coef2
            return Polynomial(new_coef[::-1])
        else:
            raise NotImplementedError

    def __rmul__(self, o):
        return self * o

    def __pow__(self, power):
        if isinstance(power, float):
            if not power.is_integer():
                raise NotImplementedError
            return self ** int(power)
        elif isinstance(power, int):
            if power < 0:
                raise NotImplementedError
            elif power == 0:
                return Polynomial([1])
            else:
                p = 1
                for _ in range(power):
                    p *= self
                return p
        elif isinstance(power, Fraction):
            if power.denominator != 1:
                raise NotImplementedError
            return self ** int(power.numerator)
        else:
            raise NotImplementedError

    def __call__(self, val):
        """Evaluate at X==val"""
        value = 0
        for p, coef in enumerate(self.coefficients[::-1]):
            value += coef * (val ** p)
        return value

    def __eq__(self, o):
        if isinstance(o, Polynomial):
            if len(self.coefficients) != len(o.coefficients):
                return False
            return all(self.coefficients[t] == o.coefficients[t] for t in range(len(self.coefficients)))
        elif isinstance(o, (int, float, Fraction)):
            return self == Polynomial([o])
        else:
            raise NotImplementedError

    def __truediv__(self, o):
        if isinstance(o, (int, float, Fraction)):
            if o == 0:
                raise ZeroDivisionError
            return self * (1 / o)

    def __str__(self):
        "Outputs Latex form of polynomial"
        s = ''
        for p, value in enumerate(self.coefficients[::-1]):
            if value == 0:
                if self.deg == 0:
                    s = str(value)
                continue
            elif value == 1:
                if p == 0:
                    s = '+1'
                else:
                    s = '+' + self.variable + '^{' + str(p) + '}' + s
            elif value == -1:
                if p == 0:
                    s = '-1'
                else:
                    s = '-' + self.variable + '^{' + str(p) + '}' + s
            else:
                tex_coef = value.process_tex()
                if tex_coef[0] != '-':
                    tex_coef = '+' + tex_coef
                s = tex_coef + (self.variable + '^{' + str(p) + '}') * (p > 0) + s
        s = s.replace('^{1}', '')
        if s[0] == '+':
            s = s[1:]
        return '$' + s + '$'


def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)


def find_lcm_of_list(list):
    x = reduce(lcm, list)
    return x


def find_gcd_of_list(list):
    x = reduce(math.gcd, list)
    return x


def input_polynomial_real(p_name: str = 'P(x)') -> Polynomial:
    print(colored(f'\n\nEnter polynomial {p_name}:', 'green'))
    deg = int(input('Enter a degree of a polynomial:'))
    assert deg >= 0, 'Degree can`t be negative!'
    coefs = []
    for power in range(deg, -1, -1):
        coef = input(f'Coefficient x^{power}:    ')
        coef = coef.replace(' ', '')
        coef = re.findall(r"[-+]?\d+", coef)
        if len(coef) == 1:
            coef = int(coef[0])
        elif len(coef) == 2:
            coef = Fraction(int(coef[0]), int(coef[1]))
        coefs += [coef]
    return Polynomial(coefs)


def input_polynomial_complex(p_name: str = 'P(x)') -> Polynomial:
    ...


def multiply_polynomials():
    c = input(colored('Use complex numbers? (+/-):  ', 'cyan'))
    while not c in '+-':
        c = input(colored('Wrong value!', 'red') + colored(' Use complex numbers? (+/-):  ', 'cyan'))
    c = (c == '-')
    # Enter P(x)
    if c:
        P = input_polynomial_real('P(x)')
    else:
        P = input_polynomial_complex('P(x)')

    # Enter Q(x)
    if c:
        Q = input_polynomial_real('Q(x)')
    else:
        Q = input_polynomial_complex('Q(x)')

    print('\n\n' + colored('Latex representation: ', 'blue'))
    print(f'({P})({Q}) = {P * Q}')


def single_long_division():
    c = input(colored('Use complex numbers? (+/-):  ', 'cyan'))
    while not c in '+-':
        c = input(colored('Wrong value!', 'red') + colored(' Use complex numbers? (+/-):  ', 'cyan'))
    c = (c == '-')
    # Enter P(x)
    if c:
        P = input_polynomial_real('P(x)')
    else:
        P = input_polynomial_complex('P(x)')

    # Enter Q(x)
    if c:
        Q = input_polynomial_real('Q(x)')
    else:
        Q = input_polynomial_complex('Q(x)')

    print('\n\n' + colored('Latex representation: ', 'blue'))
    quotient, remainder = perform_div(P, Q)
    s = (f'{P} = ' + ('(' * (not quotient.is_number()) + str(quotient) + ')' * (not quotient.is_number())) * (quotient != 1) + '(' + str(Q) + ') + ' + str(remainder)).replace('+ $-', '- $')
    print(s)
    print('Quotient: ' + str(quotient))
    print('Remainder: ' + str(remainder))
    print('Integer quotient:  ' + str(quotient * find_lcm_of_list(list(map(lambda x: x.get_common_denominator(), quotient.coefficients)))) / find_gcd_of_list(list(map(lambda x: x.get_numerator_gcd(), quotient.coefficients))))
    print('Integer remainder:  ' + str(remainder * find_lcm_of_list(list(map(lambda x: x.get_common_denominator(), remainder.coefficients)))) / find_gcd_of_list(list(map(lambda x: x.get_numerator_gcd(), remainder.coefficients))))


def perform_div(P: Polynomial, Q: Polynomial):
    """Returns: quotient, remainder"""
    if P.deg < Q.deg:
        return Polynomial([0]), P
    q_coefs = [0] * (P.deg - Q.deg + 1)
    T = P.copy()
    for t in range(len(q_coefs)):
        if T.deg + t < P.deg:
            continue
        val = T.coefficients[0] / Q.coefficients[0]
        q_coefs[t] = val
        T -= Q * val * Polynomial([1, 0]) ** (T.deg - Q.deg)
    return Polynomial(q_coefs), T


def polynomial_gcd():
    c = input(colored('Use complex numbers? (+/-):  ', 'cyan'))
    while not c in '+-':
        c = input(colored('Wrong value!', 'red') + colored(' Use complex numbers? (+/-):  ', 'cyan'))
    c = (c == '-')
    # Enter P(x)
    if c:
        first = input_polynomial_real('P(x)')
    else:
        first = input_polynomial_complex('P(x)')

    # Enter Q(x)
    if c:
        second = input_polynomial_real('Q(x)')
    else:
        second = input_polynomial_complex('Q(x)')

    if first.deg >= second.deg:
        P = first
        R = second
    else:
        P = second
        R = first

    while R != 0 and P.deg > 0:
        quotient, remainder = perform_div(P, R)
        print((f'{P} = ' + ('(' * (not quotient.is_number()) + str(quotient) + ')' * (not quotient.is_number())) * (quotient != 1) + '(' + str(R) + ') + ' + str(remainder)).replace('+ $-', '- $').replace(' + $0$', ''))
        # Integer
        if remainder != 0:
            remainder = remainder * find_lcm_of_list(list(map(lambda x: x.get_common_denominator(), remainder.coefficients)))
            remainder = remainder / find_gcd_of_list(list(map(lambda x: x.get_numerator_gcd(), remainder.coefficients)))
        P = R
        R = remainder
        if R.coefficients[0].real < 0:
            R *= -1

    P = P * find_lcm_of_list(list(map(lambda x: x.get_common_denominator(), P.coefficients)))
    P = P / find_gcd_of_list(list(map(lambda x: x.get_numerator_gcd(), P.coefficients)))
    if P.coefficients[0].real < 0:
        P *= -1
    print(f'GCD<{first}, {second}> = {P}')




if __name__ == '__main__':
    while True:
        try:
            ops = {1: multiply_polynomials, 2: single_long_division, 3: polynomial_gcd}
            print(colored('List of operations:', 'cyan'))
            print('1: Polynomial multiplication ' + colored('P(x) * Q(x)', 'green'))
            print('2: Polynomial long division: ' + colored('P(x) / Q(x)', 'green'))
            print('3: GCD of polynomials: ' + colored('GCD<P(x), Q(x)>', 'green'))

            operation = input('\n' + colored('Choose operation:   ', 'green'))
            operation = int(operation)
            if not operation in ops.keys():
                print(colored('No operation with that number', 'red'), '\n\n\n')
                continue

            # Execute operation
            ops[operation]()
        except Exception:
            print(colored('Something went wrong!', 'red'))


