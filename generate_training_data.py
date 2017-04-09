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

    i = 0
    for k in User_age_group:
        i += 1
        if i % 1000 == 0:
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


BookUsersRead = {}
BookUsersLike = {}
BookUsersDislike = {}

User2Reads = {}
User2Likes = {}
User2Dislikes = {}


def loadBookUsersRead():
    print "==== load read book users ===="
    query_statement = """SELECT `ISBN`, `User-ID` FROM `bx-book-ratings`"""
    x = conn.cursor()
    x.execute(query_statement)
    results = x.fetchall()

    print "==== store read book users into dict ===="
    for result in results:
        isbn = result[0]
        user_id = result[1]
        if isbn not in BookUsersRead:
            BookUsersRead[isbn] = set()
        BookUsersRead[isbn].add(user_id)

        if user_id not in User2Reads:
            User2Reads[user_id] = set()
        User2Reads[user_id].add(isbn)


def loadBookUsersLike():
    print "==== load like book users ===="
    query_statement = """SELECT ISBN, `User-ID` FROM `bx-book-ratings` WHERE `Book-Rating` >= 8"""
    x = conn.cursor()
    x.execute(query_statement)
    results = x.fetchall()

    print "==== store like book users into dict ===="
    for result in results:
        isbn = result[0]
        user_id = result[1]
        if isbn not in BookUsersLike:
            BookUsersLike[isbn] = set()
        BookUsersLike[isbn].add(user_id)

        if user_id not in User2Likes:
            User2Likes[user_id] = set()
        User2Likes[user_id].add(isbn)


def loadBookUsersDislike():
    print "==== load dislike book users ===="
    query_statement = """SELECT ISBN, `User-ID` FROM `bx-book-ratings` WHERE `Book-Rating` <= 3"""
    x = conn.cursor()
    x.execute(query_statement)
    results = x.fetchall()

    print "==== store dislike book users into dict ===="
    for result in results:
        isbn = result[0]
        user_id = result[1]
        if isbn not in BookUsersDislike:
            BookUsersDislike[isbn] = set()
        BookUsersDislike[isbn].add(user_id)

        if user_id not in User2Dislikes:
            User2Dislikes[user_id] = set()
        User2Dislikes[user_id].add(isbn)


def findUsersReadBook(book_isbn, user_category):
    """
    find users who read a book.
    user category:
    'read'
    'like', i.e., score >= 8
    'dislike', i.e., score <= 3
    """
    if user_category == 'read' and book_isbn in BookUsersRead:
        return BookUsersRead[book_isbn]
    if user_category == 'like' and book_isbn in BookUsersLike:
        return BookUsersLike[book_isbn]
    if user_category == 'dislike' and book_isbn in BookUsersDislike:
        return BookUsersDislike[book_isbn]
    return set()


def saveBookAgeIndications(book_isbn, user_category):
    """
    save book-age indication probabilities into DB
    user category:
    'read'
    'like', i.e., score >= 8
    'dislike', i.e., score <= 3
    """
    users = findUsersReadBook(book_isbn, user_category)
    total_user_num = 0
    user_num_age_group = [0, 0, 0, 0, 0]
    for user in users:
        if user in User_age_group:
            total_user_num += 1
            user_num_age_group[User_age_group[user] - 1] += 1.0

    user_prob_age_group = [0, 0, 0, 0, 0]
    if total_user_num > 0:
        for i in range(5):
            user_prob_age_group[i] = user_num_age_group[i] * 1.0 / total_user_num

    table_name = "`ly-book-age-{}`".format(user_category)
    insert_statement = "INSERT INTO {} (ISBN, age1, age2, age3, age4, age5, user_count) VALUES (%s, %s, %s, %s, %s, %s, %s)".format(
        table_name)
    x = conn.cursor()
    x.execute(insert_statement, (book_isbn,) + tuple(user_prob_age_group) + (total_user_num,))


def saveBooksAgeIndications(books_isbn, user_category):
    print "=====", user_category, "====="
    i = 0
    for book_isbn in books_isbn:
        saveBookAgeIndications(book_isbn, user_category)
        i += 1
        if i % 1000 == 0:
            print i
    try:
        conn.commit()
    except Exception:
        conn.rollback()


def saveBooksReadAgeInd(books_isbn):
    """
    save book-age indications for a list of books considering users' 'read' actions
    """
    saveBooksAgeIndications(books_isbn, 'read')


def saveBooksLikeInd(books_isbn):
    """
    """
    saveBooksAgeIndications(books_isbn, 'like')


def saveBooksDislikeInd(books_isbn):
    """
    save book-age indications for a list of books considering users' 'like' actions (score <= 3)
    """
    saveBooksAgeIndications(books_isbn, 'dislike')


def getBookAgeIndications(book_isbn, user_category):
    """
    read book-age indication probabilities from DB
    user category:
    'read'
    'like', i.e., score >= 8
    'dislike', i.e., score <= 3
    """
    table_name = "`ly-book-age-{}`".format(user_category)
    query_statement = "SELECT age1, age2, age3, age4, age5, user_count FROM {} WHERE ISBN = %s".format(table_name)
    x = conn.cursor()
    x.execute(query_statement, (book_isbn,))
    result = x.fetchone()
    if result is None:
        return [0.0] * 6
    else:
        return [float(i) for i in result]


def getUserBooks(user_id, user_category):
    if user_category == 'read' and user_id in User2Reads:
        return User2Reads[user_id]
    elif user_category == 'like' and user_id in User2Likes:
        return User2Likes[user_id]
    elif user_category == 'dislike' and user_id in User2Dislikes:
        return User2Dislikes[user_id]
    return set()


def aggregateBookAgeIndFeatures(user_id, user_category, aggregate_method):
    """
    aggregation_method: certain method that can aggregate a list of book age indications from
    one category (read, like or dislike) into a feature vector
    """
    books = getUserBooks(user_id, user_category)
    books_age_indications = {}
    for book in books:
        indications = getBookAgeIndications(book, user_category)
        if indications[5] > 0:  # if this book has indications in the DB
            books_age_indications[book] = getBookAgeIndications(book, user_category)
    return aggregate_method(books_age_indications)


def aggregation_method_avg(books_age_indications):
    """
    direct average to output feature vecotr
    """
    features = [0.0] * 5
    if books_age_indications is None or len(books_age_indications) == 0:
        return features
    num_books = len(books_age_indications)
    for book in books_age_indications:
        indications = books_age_indications[book]
        features = [(features[i] + indications[i] * 1.0 / num_books) for i in range(5)]
    return features


# aggregation_weighted_avg; max; min; etc.


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


def selectBooks(reader_num_threshold):
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


def outputOneLine(age_group, feature_vecs):
    line = str(age_group)
    for feature_vec in feature_vecs:
        feature_vec_str = ' '.join([str(i) for i in feature_vec])
        line += ' ' + feature_vec_str
    return line


def outputTrainingDatasets(aggregate_method, file_name):
    """
    Construct the file storing the y and X for predictive modeling
    y: age group
    X: extracted features
    """
    with open("training_data/{}.txt".format(file_name), 'w') as outputfile:
        for u in User2Reads:
            if u not in User_age_group:  # for users without age, directly passby
                continue
            age_group = User_age_group[u]
            feature_read = aggregateBookAgeIndFeatures(u, 'read', aggregate_method)
            feature_like = aggregateBookAgeIndFeatures(u, 'like', aggregate_method)
            feature_dislike = aggregateBookAgeIndFeatures(u, 'dislike', aggregate_method)
            line = outputOneLine(age_group, [feature_read, feature_like, feature_dislike])
            outputfile.write("{}\n".format(line))


def loadAll():
    loadUserAge()
    loadBookUsersRead()
    loadBookUsersLike()
    loadBookUsersDislike()


loadAll()
outputTrainingDatasets(aggregation_method_avg, 'feature_avg')


# save to database
# books_isbn = selectBooks(reader_num_threshold=10)
# saveBooksReadAgeInd(books_isbn)
# saveBooksLikeInd(books_isbn)
# saveBooksDislikeInd(books_isbn)