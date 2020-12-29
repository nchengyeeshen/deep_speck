import unittest

from speck import gen_quartet_plain


class TestGenQuartetPlain(unittest.TestCase):
    def setUp(self):
        self.diff1 = 64, 0
        self.diff2 = 127, 0
        (
            self.plain1_left,
            self.plain1_right,
            self.plain2_left,
            self.plain2_right,
            self.plain3_left,
            self.plain3_right,
            self.plain4_left,
            self.plain4_right,
        ) = gen_quartet_plain(1_000, self.diff1, self.diff2)

    def test_p1_xor_p2_equal_p3_xor_p4(self):
        message = (
            "Plaintext 1 XOR plaintext 2 should equal to plaintext 3 XOR plaintext 4"
        )
        self.assertTrue(
            all(
                self.plain1_left ^ self.plain2_left
                == self.plain3_left ^ self.plain4_left
            ),
            msg=message,
        )
        self.assertTrue(
            all(
                self.plain1_right ^ self.plain2_right
                == self.plain3_right ^ self.plain4_right
            ),
            msg=message,
        )

    def test_p1_xor_p2_equal_diff1(self):
        message = "Plaintext 1 XOR plaintext 2 should give difference 1"
        self.assertEqual(
            (self.plain1_left ^ self.plain2_left)[0], self.diff1[0], msg=message
        )
        self.assertEqual(
            (self.plain1_right ^ self.plain2_right)[1], self.diff1[1], msg=message
        )

    def test_p2_xor_p4_equal_p1_xor_p3(self):
        message = (
            "Plaintext 2 XOR plaintext 4 should equal to plaintext 1 XOR plaintext 3"
        )
        self.assertTrue(
            all(
                self.plain2_left ^ self.plain4_left
                == self.plain1_left ^ self.plain3_left
            ),
            msg=message,
        )
        self.assertTrue(
            all(
                self.plain2_right ^ self.plain4_right
                == self.plain1_right ^ self.plain3_right
            ),
            msg=message,
        )

    def test_p2_xor_p4_equal_diff2(self):
        message = "Plaintext 2 XOR plaintext 4 should give difference 2"
        self.assertEqual(
            (self.plain2_left ^ self.plain4_left)[0], self.diff2[0], msg=message
        )
        self.assertEqual(
            (self.plain2_right ^ self.plain4_right)[0], self.diff2[1], msg=message
        )
