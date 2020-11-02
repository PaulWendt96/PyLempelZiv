import os
import functools as ft
from bitarray import bitarray
import sys

def matching_characters(s1, s2):
    '''
    This returns the number of matching characters between s1 and s2. 

        matching_characters('hello', 'hell') = 4
        matching_characters('abrocad', 'abddrmi') = 2
        matching_characters('same string but starting with an s',
                            'rame string but starting with a r') = 0
        matching_characters('', 'empty') = 0
    '''
    if s1 and s2 and s1[0] == s2[0]:
        return 1 + matching_characters(s1[1:], s2[1:])
    else:
        return 0

class LempelZiv(object):
    
    max_window_size = 255 # allows us to set fixed # bits for copy dist
    def __init__(self, lookahead_size = 20, window_size = 100):
        self.lookahead_size = lookahead_size
        self.window_size = max(self.max_window_size, window_size)
        self.max_copy_length = min(15, lookahead_size)
        self.pointer_bits = 1 + 4 + 8  # signal + cp length + cp dist 
        self.literal_bits = 1 + 8  # signal + literal encoding

    def longest_substring(self, window, string):
        '''
        (copy distance, copy length) 

        This finds the copy length and copy position for a string
        in a given window of strings

        In Lempel-Ziv 77, copy length and copy position refer to data
        used to copy characters from a previously encountered string
        This allows us to construct, in effect, pointers to parts of
        the data that have been previously processed. 

        For instance, let's say we have the following window and string:
            window = 'aabbccabcabcde'
            string = 'cabcdadfbasd'
        In this case, the longest_substring function will return (5, 4)
            5, the copy position, represents the position in the window
            where the first character of the largest substring match
            can be found, e.g.

              a a b b c c a b c ...
              0 1 2 3 4 5 6 7 8

            4 refers to the copy length, which tells the decompressor
            how many characters to copy. In this case, the decompressor
            will determine that we need to copy from character 5 to
            character 8, inclusive.

        Note that, while longest_substring returns only the largest
        substring match, there are other potential character matches
        as well. The character 'c', for instance, can also be found at
        position 4 (although this match is only of length 1 since the
        following character in position 5, another 'c', doesn't
        match the following character in the string, which is 'a'
        '''

        window_range = range(self.window_size)

        matches = ((i, matching_characters(window[j:], string))
                    for i, j in enumerate(window_range))

        try:
            return max(matches, key = lambda x: x[1])
        except ValueError: # There must be a better way to handle this
            return (0, 0) 

    def make_literal(self, bitarr, char):
        bitarr.append(False)
        bitarr.frombytes((char).to_bytes(1, byteorder = 'big'))

    def make_pointer(self, bitarr, copy_distance, copy_length):
        def lengther():
            '''
            To clarify: You're looking at a really dumb
            (but neccessary) shortcut here.
            
            Python does not play nicely with the bitarray package
            if you're using a number of bits that is not a multiple of 8
            '''
            values = [True if bit == '1' else False
                           for bit in bin(copy_length)[2:]]
            pad = 4 - len(values)
            for p in range(pad):
                bitarr.append(False)
            for value in values:
                bitarr.append(value)

        bitarr.append(True)
        bitarr.frombytes((copy_distance).to_bytes(1, byteorder = 'big'))
        lengther()

    def literal_map(self, bitarr, bytestring):
        for byte in bytestring:
            self.make_literal(bitarr, byte)

    def encode(self, inPath, outPath):
        '''
        This is a (very) simplified version of the LZ77 encoder proposed
        by Abraham Lempel and Jacob Ziv in 1977.  The key idea in LZ77
        is that we can have pointers to phrases in a file that are
        repeated earlier in the file to save memory.

        The decoder can then copy the phrases from the prior point in
        the file and insert them.

        This encoder is meant to be memory efficient. It reads the
        input file in chunks as opposed to all at once.

        To implement LZ77, we need to define two structures:

            A "window_buffer", which maintains a running string of the
            prior n characters. 

            A "lookahead_buffer", which maintains a running string of
            the next m characters to process.

        LZ77 begins by defining the window buffer and lookahead buffer.
        In my implementation, the window buffer is also written to the
        bitarray as literal characters, although I can imagine
        implementations where this might not be the case

        The encoder then looks for the largest substring in the window
        buffer that matches the lookahead buffer (note that the
        lookahead buffer string starts from position zero). 

        If there was no largest substring found, the position zero
        character in the lookahead buffer is encoded as a literal in
        the bitarray. The window buffer and lookahead buffers are then
        both updated by one character.

        If there was a substring found, a pointer is made to the point
        in the lookahead buffer where the string can be found. This
        pointer defines a copy_distance and copy_length.
        The copy_distance tells the decoder how far back in the
        lookahead it should go to find the string, and the copy_length 
        tells the decoder how many characters it should copy.
        These are 8 and 4 bit integers in this implementation,
        respectively. The window and lookahead buffers are then moved
        forward copy_length characters.

        This encoder has the following characteristics:
            The cost of encoding a literal is 9 bits
                The first bit serves as a "flag" bit
                (letting the decoder know that the next 8 bits 
                represent a literal), and the next 8 bits represent
                the character itself).
                Note that the flag bit for a literal is 0.
            The cost of encoding a pointer is 13 bits
                The first bit serves as a "flag" bit
                (letting the decoder know that the next 12 bits
                represent a pointer). The next 8 bits represent
                the copy_distance, and the 4 bits
                after that represent the copy_length.

        Note that, because the cost of encoding a single literal is
        less than the cost of pointing to that literal,
        I opted to only encode pointers if copy_length is > 1.
        This conserves space. Allowing one-character 
        pointers would be wasteful.

        Obviously there were a number of trade-offs made here,
        particularly in terms of bit representation.
        For instance, representing the copy_lenth as 4 bits is limiting
        since it only allows for a maximum of 15 characters to be copied
        at a time. Adding more bits might be an improvement; however, 
        it could also be a detriment if there are only a few longer runs
        of characters.

        Note that, in general, this encoder (and, indeed, all LZ77
        encoders) will not work well with data that is loosely
        structured. Specifically, LZ77 isn't the right compression
        technique for statistical datasets, since character patterns in 
        numeric data are likely to be less frequent. LZ77 works best for
        data in which characters are structured and likely to repeat
        multiple times within a short window.
        '''

        with open(inPath, 'rb') as file:

            bitarr = bitarray(endian = 'big')
            window_buffer = file.read(self.window_size) 
            window_size = self.window_size
            self.literal_map(bitarr, window_buffer)

            lookahead_buffer = b''
            lookahead_size = 0

            # make the character in the initial buffer literals
            while True:
            
                # reset lookahead size and refill lookahead window 
                # these lines run even if file is exhausted
                #(since there may still be chars in lookahead buffer)
                out = file.read(self.lookahead_size - lookahead_size)
                if out != b'':
                    lookahead_buffer += out
                    lookahead_size = self.lookahead_size

                #do the encoding
                copy_distance, copy_length = self.longest_substring(window_buffer,
                                                                    lookahead_buffer)
                copy_distance = window_size - copy_distance
                copy_length = min(copy_length, self.max_copy_length)

                if copy_length > 1:
                    self.make_pointer(bitarr, copy_distance, copy_length)
                else:
                    copy_length = 1
                    self.make_literal(bitarr, lookahead_buffer[0])

                # update the window buffer
                window_buffer = window_buffer[copy_length:] + lookahead_buffer[:copy_length]

                #update the lookahead buffer
                lookahead_buffer = lookahead_buffer[copy_length:]
                lookahead_size -= copy_length

                #check stop condition
                if lookahead_buffer == b'':
                    bitarr.fill()
                    with open(outPath, 'wb') as outfile:
                        outfile.write(bitarr.tobytes())
                        print('File compressed and saved here: {}'.format(outPath))
                    break


    def decode (self, inFile, outFile):
        '''
        LZ77 decoder. It operates on the following rules:
            1. If the current bitarray has a sufficient number (>= 9)
               of bits remaining:
                a. Pop the first bit from the array
                b. If the first bit is a zero:
                    i. Write the corresponding decoded character to output buffer (8 bits)
                    ii. Remove 8 bits from bit stream (e.g. go to next string of bits)
                c. Otherwise,
                    i. Get the copy distance and copy pointer
                    ii. Determine the string to copy from the output buffer and copy it
                    iii. Append the copy string to the output buffer.
                    iv. Remove 13 bits from bit stream (e.g. go to next string of bits)
        '''

        def length_decode(copy_length):
            '''
            This turns a string of bits into a copy length.
            A custom method is needed for this since (to the extent of
            my knowledge) either python or the bitarr module really
            struggle supporting 4-bit integers.

                length_decode(b'1011') = 11
                length_decode(b'0001') = 1
                length_decode(b'0101') = 5

            '''
            total = 0
            for c in copy_length:
                total *= 2
                if c:
                    total += 1
            #total = int(copy_length.to01(), 2)
            return total

        outBuffer = b''
        bitarr = bitarray(endian = 'big')
        with open(inFile, 'rb') as infile:
            bitarr.fromfile(infile)

        counter = 0 # DELETE

        while len(bitarr) >= 9:
            counter += 1 # DELETE
            if counter % 1000 == 0: # DELETE
                print('{} bits left...'.format(len(bitarr))) # DELETE
            flag = bitarr[0]
            if not flag:
                byte = bitarr[1:9].tobytes()
                outBuffer += byte
                del bitarr[0:9]
            else:
                copy_distance = int.from_bytes(bitarr[1:9].tobytes(), 'big')
                copy_length = bitarr[9:13]
                copy_length = length_decode(copy_length)
                if copy_distance == copy_length:
                    copy_string = outBuffer[-copy_distance:]
                else:
                    copy_string = outBuffer[-copy_distance:
                                            -copy_distance + copy_length]
                outBuffer += copy_string
                del bitarr[0:13]

        with open(outFile, 'w+') as outfile:
            outfile.write(outBuffer.decode('utf-8'))

        print('File successfully decompressed.')

if __name__ == '__main__':
    type = sys.argv[1]
    file = sys.argv[2]
    out = sys.argv[3]
    compressor = LempelZiv()
    if type == 'compress':
        compressor.encode(file, out)
    elif type == 'decompress':
        compressor.decode(file, out)
    else:
        print('Error: Compression option must be either compress or decompress')
