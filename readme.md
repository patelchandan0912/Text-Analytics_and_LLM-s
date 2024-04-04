# Text Codecs Introduction

**Week01, Section 01**

--------

## Introduction:

**Two File Types: Text and Binary**

In the context of programming, we can broadly categorize files into two types: text files, which hold data that is human-readable, and binary files, containing data that isn't intended for direct human interpretation. Our main emphasis throughout this course will be text files.

**Operating system only see's bytes**

It's crucial to note that, for an operating system, all files, whether text or binary, are merely sequences of bytes stored on a disk. The operating system does not possess the ability to understand the significance of these bytes. The onus falls upon the application to interpret these bytes correctly. This necessitates the application's awareness of the bytes' encoding, ensuring their accurate interpretation as intelligible text.

**Character Encoding and Decoding**

Character encodings denote the mappings between raw binary data (comprised of 0's and 1's) and text characters. If a text file encoded using a certain encoding is decoded using a different encoding, it can result in the transformation of the output text, sometimes to the point of illegibility.

**Encoding/Decoding is Application Dependent**

The encoding of a file is defined by the application creating the file. During both file creation and file reading, the application must clearly specify the encoding.

**Encoding Standards Exist**

Fortunately, there exist standard (and some non-standard) encoding techniques. ASCII, UTF-8, and UTF-16 are among the most prevalently used encoding techniques, enabling representation of a vast character set and promoting interoperability among varying systems and languages.

**Understanding encoding/decoding of text files is important**

For developers and programmers, understanding the concept of file encoding is vital, as it plays an indispensable role in ensuring that data, when working with text files, is interpreted and displayed correctly.


## Section Objectives:

This section aims to equip participants with the ability to:

* Comprehend how computers perceive data: binary, hexadecimal, bytes, little endian, and big endian
* Understand the notion of encoding and its significance
* Distinguish between a byte and a character
* Grasp the concept of character/file encoding and character mapping.
* Familiarize with the most common character encodings
* Differentiate between ASCII, Unicode (UTF-8, UTF-16, UTF-32), and ISO-8959-x codepages
* Comprehend the difference between Codepage and UTF encoding

## How Computers Perceive Data:

**Binary**
Computers store all data in a binary format, indicating that data is represented as sequences of 0's and 1's, the only format that computers can process.

**Bits and Bytes**
The smallest unit of data a computer can process is a bit, which can either be a 0 or a 1. A cluster of 8 bits is known as a byte, capable of representing 256 unique values (2^8). In a computer, a byte represents the smallest addressable unit of memory.

**Hexidemical**
A byte can be depicted in hexadecimal format. Hexadecimal is a base-16 number system that uses the digits 0-9 and the letters A-F to represent the values 0-15. Hexadecimal is widely employed in computing, given its more readable format compared to binary. Each hexadecimal digit represents 4 bits (2^4 = 16). Consequently, two hexadecimal digits can represent a byte (2^8 = 256).

The following table shows the decimal, binary, and hexadecimal representations of the numbers 0-15.

| Decimal | Binary | Hexadecimal |
|---------|--------|-------------|
| 0       | 0000   | 0           |
| 1       | 0001   | 1           |
| 2       | 0010   | 2           |
| 3       | 0011   | 3           |
| 4       | 0100   | 4           |
| 5       | 0101   | 5           |
| 6       | 0110   | 6           |
| 7       | 0111   | 7           |
| 8       | 1000   | 8           |
| 9       | 1001   | 9           |
| 10      | 1010   | A           |
| 11      | 1011   | B           |
| 12      | 1100   | C           |
| 13      | 1101   | D           |
| 14      | 1110   | E           |
| 15      | 1111   | F           |

Since the smallest unit of memory is a byte, each of these values would actually be represented as bytes. For example, the number 10 would be represented as 00001010 in binary and 0A in hexadecimal.

The following table shows the decimal, binary, and hexadecimal representations of the numbers 0-255 as bytes.

| Decimal | Binary | Hexadecimal |
|---------|--------|-------------|
| 0       | 00000000   | 00           |
| 1       | 00000001   | 01           |
| 2       | 00000010   | 02           |
| 3       | 00000011   | 03           |
| 4       | 00000100   | 04           |
| 5       | 00000101   | 05           |
| 6       | 00000110   | 06           |
| 7       | 00000111   | 07           |
| 8       | 00001000   | 08           |
| 9       | 00001001   | 09           |
| 10      | 00001010   | 0A           |
| 11      | 00001011   | 0B           |
| 12      | 00001100   | 0C           |
| 13      | 00001101   | 0D           |
| 14      | 00001110   | 0E           |
| 15      | 00001111   | 0F           |
| 16      | 00001111   | 10           |
| 17      | 00001111   | 11           |
|  ...    |  ...   | ...           |
| 255      | 00001111   | FF           |

### Endianness:

Some data requires multiple bytes to store the information in memory. Endianess defines the sequence in which the bytes of a multi-byte data-item are stored and accessed in memory. Two common types of endianess are little-endian and big-endian.

For instance, let's consider the hexadecimal number 0x12345678, which is a 32-bit (or 4-byte) number. We can think of this as 0x12, 0x34, 0x56, and 0x78, where each byte is represented by two hexadecimal digits. The following diagram shows the byte representation of the number 0x12345678 in both big-endian and little-endian formats.

**Big-Endian:**
In Big-Endian format, the most significant byte (MSB) is stored at the lowest memory address and the least significant byte (LSB) is stored at the highest memory address. If we consider the hexadecimal number 0x12345678, the byte representation would be as follows:

Address| 	Value|
0x0000	| 0x12|
0x0001	| 0x34|
0x0002	| 0x56|
0x0003	| 0x78|
 
**Little Endian**
In contrast, in Little-Endian format, the LSB is stored at the lowest memory address, and the MSB at the highest memory address. For the same hexadecimal number 0x12345678, the byte representation would look like this:

| Address|Value|
|---|---|
| 0x0000|0x78|
| 0x0001|0x56|
| 0x0002|0x34|
| 0x0003|0x12|


**Most modern computers are little endian.**
Most modern computer systems, including x86 and x86-64, are little-endian. However, networks typically use big-endian, also known as network byte order, to improve performance and efficiency. It's therefore crucial to have a clear understanding of these concepts, especially when working with low-level programming, computer networks, or systems that interact with hardware directly.

## Understanding Encoding:

Encoding involves the conversion of data from one format to another. Regarding text files, encoding pertains to the transformation of text characters to bytes, and vice versa.

A character encoding provides a mapping between characters and bytes, defining how characters are translated into bytes, and how bytes are interpreted as characters.

Despite there being numerous ways to map characters and bytes, the most common character encodings are ASCII, UTF-8, and UTF-16. These encodings allow us to represent a wide range of characters, facilitating interoperability between different systems and languages.

## In-depth Exploration of Text Encoding Standards

In the context of electronic communication, text encoding standards are pivotal as they define a set of characters and assign each a unique number, facilitating standardized encoding and decoding processes. Various encoding standards have been developed over the years, each with their own specifications and purposes.

### ASCII Encoding

ASCII, or the American Standard Code for Information Interchange, emerged as a seminal character encoding standard in 1963 and has withstood the test of time, being extensively used even today. With its 7-bit encoding scheme, ASCII accommodates 128 unique characters. This includes the complete set of English uppercase and lowercase letters (A-Z, a-z), the digits 0-9, a range of punctuation marks, as well as a collection of control characters instrumental in formatting and controlling the flow of text.

A crucial aspect of ASCII to note is that it is a subset of UTF-8, which makes it backward compatible with the latter. In simpler terms, this implies that all ASCII characters can be interpreted correctly by a system designed for UTF-8.

### Unicode and its Variants (UTF-8 and UTF-16)

In 1991, a more expansive encoding standard known as Unicode was introduced. Utilizing a 21-bit encoding scheme, Unicode can represent over 2 million unique characters (2^21 = 2,097,152), far exceeding the capacity of ASCII. Unicode includes characters from a plethora of languages worldwide, making it a comprehensive global standard.

Two commonly used forms of Unicode are UTF-8 and UTF-16. Both variants were first published in 1993, defining the same character set as Unicode. However, they differ in the byte-length of their encoding.

UTF-8 employs a variable-length encoding scheme, where each character can be represented by 1-4 bytes. This adaptability makes UTF-8 efficient for a wide range of use-cases, and particularly suited to languages where most characters fall within the ASCII range, as these can be represented using just a single byte. UTF-8 is backward compatible with both ASCII and Unicode, which means ASCII characters can be read accurately by UTF-8 systems, and UTF-8 characters by Unicode systems.

Conversely, UTF-16 also uses a variable-length encoding scheme but each character is represented by either 2 or 4 bytes, depending on the character. This is generally more efficient for scripts (like many East Asian scripts) where most characters fall outside of the ASCII range. UTF-16 maintains backward compatibility with ASCII and Unicode as well.

### Codepages and ISO-8859-x

Codepages are another type of character encoding standards that gained popularity around 1987. Unlike Unicode that caters to a vast array of languages with a single comprehensive standard, codepages cater to different languages through individual standards, each representing a unique set of characters with a fixed number of bytes.

A common family of codepages is ISO-8859-x. This family employs an 8-bit encoding scheme and can represent 256 unique characters (2^8 = 256), including the English alphabet, digits, punctuation marks, control characters, and characters from various other European languages. It is backward compatible with ASCII and Unicode.

---

### Application of Encoding in Python

In Python, the encoding plays a vital role when dealing with file I/O operations. The encoding standard must be specified when reading or writing a file. If not explicitly stated, Python uses the default encoding, which is platform-dependent (e.g., cp1252 on Windows, utf-8 on Linux).

### Reading a File with Specific Encoding

To read a file with a specific encoding, the 'encoding' parameter of the open() function is used. Here's an example:

```python
with open('file.txt', encoding='utf-8') as f:
    data = f.read()
```

In this snippet, the file 'file.txt' is opened using the UTF-8 encoding, and its contents are read into the variable 'data'.

### Writing to a File with Specific Encoding

Similarly, when writing to a file, the encoding should be stated explicitly. If not, Python will use the platform's default encoding. Here's how to specify the encoding:

```python
with open('file.txt', 'w', encoding='utf-8') as f:
    f.write('Hello, world!')
```

In this snippet, the string 'Hello, world!' is written to 'file.txt' using UTF-8 encoding.

### Managing Encoding Errors

Sometimes, encoding errors can occur if the selected encoding doesn't match the file's actual encoding. This will result in a UnicodeDecodeError, indicating that the file could not be decoded. For instance, if a file encoded with binary data was attempted to be read with UTF-8 encoding, Python would raise an error as shown below:

```python
with open('./data/binary.bin', 'r', encoding='utf-8') as f:
    print(f.read())
```

This will lead to a traceback ending with a UnicodeDecodeError, demonstrating an issue with a specific byte in the file which could not be decoded using the chosen encoding.

---

> Remember, understanding and correctly applying text encoding standards is a cornerstone of managing and manipulating text data in any programming language, and Python is no exception. By mastering these concepts, students will gain a robust foundation in data handling, which is integral to numerous fields, including data analytics, web development, artificial intelligence, and many more.


## How to detect the encoding of a file

When reading a file, the encoding must be specified. If the encoding is not specified, the default encoding will be used. The default encoding is platform dependent. For example, on Windows the default encoding is cp1252, while on Linux the default encoding is utf-8. With Python programming, the default encoding is UTF-8.

Ideally, you will be given information on how the file your are processing is encoded. If you do not have this information you have a couple different approaches:
1) Try to read the file using different encodings until you find one that works.
2) Use a library to detect the encoding of the file (one popular one in Python is chardet)
3) Use a hex editor to view the file and try to determine the encoding.

> ### Note About Glyphs
>
> A glyph is a graphical representation of a character. A glyph can be a single character, a sequence of characters, or a combination of characters and symbols. Glyphs are used to represent text in a variety of languages, including English, Chinese, Japanese, Korean, and Arabic.
>
>When we are working with text data, we are concerned with the correct intrpretation of what the data represents (i.e. the characters). The glyphs are the visual representation of the characters. Generally, we are not concerned with the glyphs, but rather the characters that the glyphs represent. 

## Conclusion

In this notebook, we learned about encoding and decoding text files. Despite the complexity of what we seen, the basic idea is simple. A text file is a sequence of bytes. The encoding scheme tells us how to interpret the bytes in the file. The decoding scheme tells us how to convert the bytes in the file to characters. The encoding scheme and the decoding scheme must match. If they do not match, we will get an error.

In most cases; you will not have to worry about encoding and decoding - the default of utf-8 will work. However, there are times when issues will arise, and you need to know how to fix any issues. If you have an understanding of the concepts covered in this notebook, you will be in much better position to troubleshoot and fix any issues that arise.

A suggest approach is:
1) Try loading file/data with default encoding (utf-8)
2) If an error, is it related to encoding
   *) I've included examples of these errors in the code above
2) Do you know the possible encoding that might be used?
   *) If yes, try loading file/data with that encoding
   *) Common alternatives to Unicode (utf-8, utf-16 utf-32) are Windows-1252, MacRoman, ISO-8859-1
3) Sometimes, you can load a file using one codepage that was created in another and not see an error.
   *) If there are special characters used in the file, this will most likely result in these characters being displayed incorrectly.
4) If you still haven't resolved the problem; try using the model chardet 
   *) Chardet will attempt to determine the encoding of a file. It is not perfect, but it can help you determine the encoding of a file.
5) If you still haven't resolved the problem; try using the model ftfy (https://ftfy.readthedocs.io/en/latest/)

## References

* Python Codecs (Encoding/Decoding) - [https://docs.python.org/3.11/library/codecs.html](https://docs.python.org/3.11/library/codecs.html)
* Binary, Hex, and Decimal Converter - [https://www.rapidtables.com/convert/number/ascii-to-binary.html](https://www.rapidtables.com/convert/number/ascii-to-binary.html)
* Endianness - [https://en.wikipedia.org/wiki/Endianness](https://en.wikipedia.org/wiki/Endianness)
* ASCII Table - [https://www.asciitable.com/](https://www.asciitable.com/)
* Unicode Table - [https://unicode-table.com/en/](https://unicode-table.com/en/)
  * To see how Unicode is segmented into ranges (or blocks) for languages/charactets, see here https://symbl.cc/en/unicode/blocks/
  * UTF-8 Table - [https://www.utf8-chartable.de/](https://www.utf8-chartable.de/)
  * UTF-16 Table - [https://www.fileformat.info/info/charset/UTF-16/list.htm](https://www.fileformat.info/info/charset/UTF-16/list.htm)
  * UTF-32 Table - [https://www.fileformat.info/info/charset/UTF-32/list.htm](https://www.fileformat.info/info/charset/UTF-32/list.htm)
* IS0-8859-x Table - [https://en.wikipedia.org/wiki/ISO/IEC_8859](https://en.wikipedia.org/wiki/ISO/IEC_8859)
* Chardet - [https://pypi.org/project/chardet/]
* Hex Editors - [https://en.wikipedia.org/wiki/Comparison_of_hex_editors](https://en.wikipedia.org/wiki/Comparison_of_hex_editors)
* VSCode hex Editor Extension - [https://marketplace.visualstudio.com/items?itemName=ms-vscode.hexeditor](https://marketplace.visualstudio.com/items?itemName=ms-vscode.hexeditor)
* The concept of Glyphs - [https://en.wikipedia.org/wiki/Glyph](https://en.wikipedia.org/wiki/Glyph)


## Questions:

* What is a text codec?
* What is a character encoding?
* What a glyph? And, how is it related to a character?
* What is the difference between a character and a byte?


