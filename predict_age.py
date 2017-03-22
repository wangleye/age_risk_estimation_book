"""
    This is an algorithm to predict age from users' reading preferences
    based on book crossing dataset.
    Copyright (C) 2017  Leye Wang (wangleye@gmail.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pymysql

conn = pymysql.connect(host='127.0.0.1',
                       user='root',
                       passwd='123456',
                       db='book_crossing')
User_age_group = {}  # store the users' age groups, use loadUserAge() function to initialize


def loadUserAge():
    """
    load user ages from db
    """
    query_statement = """SELECT `User-ID`, Age FROM `bx-users` WHERE Age is not NULL """
    x = conn.cursor()
    x.execute(query_statement)
    results = x.fetchall()
    for result in results:
        User_age_group[result[0]] = age2group(int(result[1]))
    for k in User_age_group:
        print k, User_age_group[k]


def age2group(age):
    """
    change age to a group number:
    totally 5 groups
    1: <= 20 yr
    2: 21~30 yr
    3: 31~40 yr
    4: 41~50 yr
    5: >= 51 yr
    """
    return max(min(int((age - 1) / 10), 5), 1)


def findUsersReadBook(book_isbn, user_category):
    """
    find users who read a book.
    user category:
    'read'
    'like', i.e., score >= 7
    'dislike', i.e., score <= 3
    """
    query_statement = """SELECT DISTINCT(`User-ID`) FROM `bx-book-ratings` WHERE ISNB = %s """
    if user_category == 'like':
        query_statement += """AND `Book-Rating` >= 7"""
    elif user_category == 'dislike':
        query_statement += """AND `Book-Rating` <= 3"""
    x = conn.cursor()
    x.execute(query_statement, (book_isbn))
    results = x.fetchall()
    users = []
    for result in results:
        users.append(result[0])
    return users


def saveBookAgeIndications(book_isbn, user_category):
    """
    save book-age indication probabilities into DB
    user category:
    'read'
    'like', i.e., score >= 7
    'dislike', i.e., score <= 3
    """
    users = findUsersReadBook(book_isbn, user_category)
    total_user_num = 0
    user_num_age_group = [0, 0, 0, 0, 0]
    for user in users:
        if user in User_age_group:
            total_user_num += 1
            user_num_age_group[User_age_group[user] - 1] += 1

    user_prob_age_group = [0, 0, 0, 0, 0]
    for i in range(5):
        user_prob_age_group[i] = user_num_age_group[i] * 1.0 / total_user_num

    table_name = "`ly-book-age-{}`".format(user_category)
    insert_statement = "INSERT INTO {} (ISBN, age1, age2, age3, age4, age5) VALUES (%s, %s, %s, %s, %s, %s)".format(
        table_name)
    x = conn.cursor()
    x.execute(insert_statement, (book_isbn,) + tuple(user_prob_age_group))


def saveBooksAgeIndications(books_isbn, user_category):
    for book_isbn in books_isbn:
        saveBookAgeIndications(book_isbn, user_category)
    x = conn.cursor()
    try:
        x.commit()
    except Exception:
        x.rollback()


def saveBooksReadAgeInd(books_isbn):
    """
    save book-age indications for a list of books considering users' 'read' actions
    """
    saveBooksAgeIndications(books_isbn, 'read')


def saveBooksLikeInd(books_isbn):
    """
    save book-age indications for a list of books considering users' 'like' actions (score >= 7)
    """
    saveBooksAgeIndications(books_isbn, 'like')


def saveBooksDislikeInd(books_isbn):
    """
    save book-age indications for a list of books considering users' 'like' actions (score <= 3)
    """
    saveBooksAgeIndications(books_isbn, 'dislike')


def saveBookCounts2db(book_isbn, book_count):
    insert_statement = """INSERT INTO `ly-book-readcount` (ISBN, Count) VALUES (%s,%s)"""
    x = conn.cursor()
    x.execute(insert_statement, (book_isbn, book_count))


def saveBookReads():
    query_statement = "SELECT ISBN, count(*) from `bx-book-ratings` group by ISBN"
    x = conn.cursor()
    x.execute(query_statement)
    results = x.fetchall()
    i = 0
    for result in results:
        book_isbn = result[0]
        book_count = result[1]
        saveBookCounts2db(book_isbn, book_count)
        i += 1
        if i % 100 == 0:
            print i
    try:
        conn.commit()
    except Exception:
        conn.rollback()


def selectBooks(reader_num_threshold=20):
    """
    select the books whose reader number is larger than a threshold
    """
    query_statement = """SELECT ISBN from `ly-book-readcount` where Count >= %s"""
    x = conn.cursor()
    x.execute(query_statement, (reader_num_threshold))
    results = x.fetchall()
    books_isbn = []
    for result in results:
        books_isbn.append(result[0])
    return books_isbn


# saveBookReads()
# selectBooks()
loadUserAge()
