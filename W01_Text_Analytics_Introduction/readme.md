# Introduction to Text Analytics


---

## 1.1 What is Text?

In the context of data and information systems, text refers to the alphanumeric characters that constitute written or printed matter. It is a form of unstructured data, which contrasts structured data that is readily quantifiable and storable in conventional databases. Text data can come from a variety of sources, including books, articles, web content, social media posts, emails, documents, and more. In essence, any written material, digital or otherwise, can be considered as text data.

---

## 1.2 What is Text Analytics?

Text analytics, also known as text data mining, is the process of deriving high-quality information from text. This process involves structuring the input text, determining patterns within the structured data, and finally interpreting the output. This is typically done via sophisticated algorithms and machine learning techniques. Text analytics is a broad field that encompasses several sub-disciplines such as sentiment analysis, topic modeling, entity extraction, and many others.

---

## 1.3 Why Text Analytics?

Text is one of the most common ways in which information is exchanged, and there is an enormous amount of text data available to us. Text analytics allows us to analyze this vast amount of data and derive insights from it. By applying text analytics, businesses can gain a competitive edge by understanding their customers better, enhancing decision-making processes, identifying trends, and predicting future scenarios. Moreover, in the era of big data, where data is becoming increasingly unstructured, the ability to analyze text data effectively is becoming more and more crucial.

---

## 1.4 Business Applications of Text Analytics

There are numerous applications of text analytics in the business sphere. Companies can utilize text analytics to analyze customer feedback and social media posts to understand customer sentiment, enabling them to react promptly to customer needs. It can also be used in market research to identify trends and patterns. In human resources, text analytics can aid in analyzing employee feedback and improving workplace environments. Furthermore, in the risk management sector, text analytics can aid in fraud detection and compliance monitoring by identifying suspicious patterns in textual data.

---

## 1.5 What are the differences between Text Mining, NLP, and Computational Linguistics?

While these terms are often used interchangeably, there are subtle differences between them. 

- Text Mining, as previously discussed, is the process of extracting high-quality information from text data. It involves the discovery of new, previously unknown information, by automatically extracting information from different written resources.

- Natural Language Processing (NLP) is a subset of artificial intelligence that focuses on the interaction between computers and humans through natural language. The goal of NLP is to read, decipher, understand, and make sense of the human language in a valuable way.

- Computational Linguistics, on the other hand, is an interdisciplinary field concerned with the computational modeling of natural language, as well as the study of computation's role in the structure of language, linguistic cognition, and language use.

---

## 1.6 Summary

Text Analytics is a powerful tool that allows businesses to derive high-quality insights from unstructured text data. With a myriad of applications ranging from customer sentiment analysis to risk management, text analytics provides a competitive advantage in the era of big data. Though text mining, NLP, and computational linguistics are closely related, they each have their unique focuses and applications.

---

## 7. Questions to Think About

1. How can text analytics enhance your current business or research processes?
2. What are the potential challenges in implementing text analytics in your field of work or study?
3. How do text mining, NLP, and computational linguistics intersect, and how do they differ in their application?
4. What potential ethical considerations should be taken into account when using text analytics?
5. How does text analytics fit into the broader landscape of data analytics and artificial intelligence?

---


# Summary Text Analytics Week 01


-----

## Encoding/Decoding Text
* Comprehend how computers perceive data: binary, hexadecimal, bytes, little endian, and big endian
* Understand the notion of encoding and its significance
* Distinguish between a byte and a character
* Grasp the concept of character/file encoding and character mapping.
* Familiarize with the most common character encodings
* Differentiate between ASCII, Unicode (UTF-8, UTF-16, UTF-32), and ISO-8959-x codepages
* Comprehend the difference between Codepage and UTF encoding

## Codecs and Python
* Utilize Python programming language for reading and writing text files with various encodings
* Implement encoding/decoding in Python
* Know how to specify the encoding during file reading and writing
* Address encoding errors effectively
* Identify the encoding of a file

## File I/O in Python
* Open a file for reading as text
* Open a file for writing as binary
* Open a file for reading and writing as text
* Open a file for reading and writing as binary
* Seeking in a file
* Reading a file line by line
* Reading a file character by character
* Reading a file all at once

## Exploratory Text Analysis
- Introduce and Review important Concepts
  - Corpus, a collection of documents (in the example below, a single document)
  - Tokenization, the process of breaking up a document into individual words
  - Lemizaton, the process of reducing words to their root form
  - Stopwords, words that are not useful for analysis
  - n-Grams, a sequence of n words
  - Wordcloud, a visualization of the most common words in a corpus
  - Sentiment Analysis, the process of determining the sentiment of a document (positive, negative, neutral)
- How to conduct exploratory textual data analysis
  - Load text
  - Tokenize
  - Filter 
    - remove stopwords
    - remove certain patterns (i.e. non-alphabetic characters)
  - Lemmatize Words
    - Not always necessary - depends on goals
  - Analysis
    - Word Frequencies
      - calculate word frequencies
      - chart word frequencies
    - Wordcloud
      - generate wordcloud from word frequencies
      - generate wordcloud from text
        - n-grams
      - adjust wordcloud appearance
        - masking (shape)
    - Sentiment
      - positive, neutral, negative and composite
      - chart sentiment



```python

```
