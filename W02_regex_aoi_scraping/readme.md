# Data Collection and Corpus Preparation, Web API's, Web Scraping


**Week02**

---

## 1.0 Introduction 

In this week, we will cover the following topics:

    * structured and semistructured data
    * regular expressions (regex)
    * web-apis
    * web-scraping
      * scrapy example

You will find a notebook for each one of these topics located in the Week02 folder.  The notebooks are numbered in the order that you should complete them.  The notebooks are:

    * W02.1-structured-semistructured-data.ipynb
    * W02.2-regular-expressions.ipynb
    * W02.3-web-apis.ipynb
    * W02.4-web-scraping.ipynb


## 3. Introduction to Web API's in Python



## Objective
The objective of this notebook is to introduce you to the concept of Web APIs and how to interact with them using Python.

## Table of Contents
1. [Introduction](#Introduction)
2. [Prerequisites](#Prerequisites)
3. [What is an API?](#What-is-an-API?)
4. [HTTP Methods: GET vs POST](#HTTP-Methods:-GET-vs-POST)
5. [Setting up the Environment](#Setting-up-the-Environment)
6. [The `requests` Library](#The-requests-Library)
7. [Making API Requests](#Making-API-Requests)
    - [GET Example: OpenWeatherMap](#GET-Example:-OpenWeatherMap)
    - [POST Example: JSONPlaceholder](#POST-Example:-JSONPlaceholder)
8. [Parsing JSON Data](#Parsing-JSON-Data)
9. [Summary](#Summary)
10. [Further Reading](#Further-Reading)

---

## Introduction
Web APIs provide a mechanism to interact with remote services, fetch data, or perform actions. They are an essential part of modern web development and data science.

---

## Prerequisites
- Basic understanding of Python
- Familiarity with JSON data structure

---

## What is an API?
API stands for Application Programming Interface. It allows different software to communicate with each other. A Web API, specifically, allows you to interact with services over the internet.

---

## HTTP Methods: GET vs POST

HTTP (HyperText Transfer Protocol) has various methods for interacting with resources, the two most common being GET and POST.

- **GET:** Fetches data from a specified resource. It's non-destructive, meaning it doesn't alter the state of the resource.
  
- **POST:** Submits data to a specified resource for processing. This can result in the creation of a new resource or updates to existing resources.

---

## Setting up the Environment

You need to install the `requests` library to make HTTP requests. You can do this using `pip` or `conda`.


## The `requests` Library

Python has several libraries for making HTTP requests, but `requests` is one of the simplest and most popular.

---

## 2.0 Making API Requests

### GET Example: OpenWeatherMap

Fetching current weather data for London.


##  Introduction to Regex Expressions

Regular expressions (regex or regexp) are patterns used to match character combinations in strings. They are a powerful tool for working with text search, manipulation, and validation. You can think of regular expressions like a "mini-language" for matching strings against patterns. They provide a powerful way to manipulate and analyze text in programs.

## Python regular expression methods


* `re.match(pattern, string, flags=0)` # match from the beginning of the string
* `re.search(pattern, string, flags=0)` # search the entire string
* `re.fullmatch(pattern, string, flags=0)` # match the entire string with the pattern

> re.match is anchored at the start ^pattern
> * Ensures the string begins with the pattern
> 
> re.fullmatch is anchored at the start and end of the pattern ^pattern$
> * Ensures the full string matches the pattern (can be especially useful with alternations as described here)
> 
> re.search is not anchored pattern
> * Ensures the string contains the pattern


* `re.findall(pattern, string, flags=0)` # find all matches in the string
* `re.sub(pattern, repl, string, count=0, flags=0)` # replace all matches in the string
* `re.split(pattern, string, maxsplit=0, flags=0)` # split string by the occurrences of pattern


* `re.compile(pattern, flags=0)` # compile a regular expression pattern into a regular expression object

> compile will save computation if you need to repeatedly use the same pattern






### Character Classes

Character classes let you define specific groups of characters to match within strings. 

   - `\d` matches any digit from 0 to 9.
   - `\w` matches any alphanumeric character including underscores.
   - `\s` matches any whitespace character such as space, tab, newline, etc.
   - `.`  matches any character 
   - These classes can help find patterns based on specific types of characters.

> Use the . operator carefully since often class or negated character class (which we’ll cover next) are faster and more precise.
>
> `\d`, `\w` and `\s` also present their negations with `\D,` `\W` and `\S` respectively.
>
> For example, `\D` will perform the inverse match with respect to that obtained with `\d`.
> 
> * `\D`         matches a single non-digit character -> Try it!
>
> In order to be taken literally, you must escape the characters ^.[$()|*+?{\with a backslash \ as they have special meaning.
>
> * `\$\d`       matches a string that has a $ before one digit -> Try it!
> Notice that you can match also non-printable characters like tabs \t, new-lines \n, carriage returns \r.

## 4. Introduction Web Scrapping

Web scraping is a technique to extract data from websites. It is also called web data extraction. It is a field with active developments sharing a common goal with the semantic web vision, an ambitious initiative that still requires breakthroughs in text processing, semantic understanding, artificial intelligence and human-computer interactions. Current web scraping solutions range from the ad-hoc, requiring human effort, to fully automated systems that are able to convert entire web sites into structured information, with limitations.



##  Web Scraping using Python

### Getting the data from websites

The protocol to access a website is called HTTP (HyperText Transfer Protocol). It is a protocol to exchange information between computers. The client (your computer) sends a request to the server (the computer where the website is hosted) and the server sends back a response. The response is usually an HTML file. HTML is a markup language to describe web pages. It contains the information displayed on the website and a lot of other information about the page. The HTML file can be parsed by the browser to display the page to the user. The HTML file can also be parsed by a computer program to extract the data from the page. This is called web scraping.

In python, there are two general/popular ways to get the data from websites:

1. Using the `requests` library

Requests library is one of the most popular libraries in Python. It is used to send HTTP requests to the server and receive a response back. It is a very powerful library and has many useful functions. It is also very easy to use. It is a must-have library for every Python developer.

2. Using selenium

Selenium is a web automation framework that can be used to automate website testing. It is open-source and can be used to automate anything we do on the web. It is also very easy to use. It is a must-have library for every Python developer. For web scraping, Selenium is often used when the website is dynamic and requires user interaction to load the data. For example, when the website uses JavaScript to load the data, we can use Selenium to automate the process of clicking the button to load the data.

3. Using Scrapy

Scrapy is a fast high-level web crawling and web scraping framework, used to crawl websites and extract structured data from their pages. It can be used for a wide range of purposes, from data mining to monitoring and automated testing. Unlike the previous two approachs, Scrapy both downloads and processes the data, saving the user from having to do it manually. 
   

### Processing the data (HTML)

After getting the data from the website, we need to process the data to extract the information we want. The data is usually in HTML format. HTML is a markup language to describe web pages. It contains the information displayed on the website and a lot of other information about the page. The HTML file can be parsed by the browser to display the page to the user. The HTML file can also be parsed by a computer program to extract the data from the page. This is called web scraping.

There are two general/popular ways to process the data:

1. Using the `BeautifulSoup` library
2. Using the `Scrapy` library

Since scrapy handles both data collection and data processing, it doesn't involve the use of BeautifulSoup or selenium. BeuatifulSoup though requires the use of requests library or selnium to get the data from the website. Beautiful soup focuses on data processing of web pages. It is a Python library for pulling data out of HTML and XML files. It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. It commonly saves programmers hours or days of work.

##  Examples using BeautifulSoup

### Example 1: Scraping a static website

In this example, we will scrape the data from a static website. A static website is a website that doesn't use JavaScript to load the data. The data is already in the HTML file. We can use the requests library to get the data from the website and then use the BeautifulSoup library to process the data.

The website we will scrape is https://www.imdb.com/chart/top. It is a list of the top 250 movies on IMDb. We will scrape the title, year, rating, and number of votes of each movie.

First, we need to import the requests library and the BeautifulSoup library.


```python

```
