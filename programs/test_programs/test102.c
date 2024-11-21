#include <stdio.h>
#include <string.h>

#define MAX_BOOKS 5
#define MAX_USERS 3
#define MAX_NAME_LEN 50

// Struct for Book
typedef struct {
    int id;
    char title[MAX_NAME_LEN];
    char author[MAX_NAME_LEN];
    int isBorrowed;
} Book;

// Struct for User
typedef struct {
    int id;
    char name[MAX_NAME_LEN];
    int borrowedBooks[MAX_BOOKS]; // List of borrowed book IDs
    int borrowedCount;
} User;

// Function prototypes
void initializeBooks(Book books[], int count);
void initializeUsers(User users[], int count);
void borrowBook(Book books[], int bookCount, User *user, int bookId);
void returnBook(Book books[], int bookCount, User *user, int bookId);
void printLibraryStatus(const Book books[], int bookCount, const User users[], int userCount);

int main() {
    Book books[MAX_BOOKS];
    User users[MAX_USERS];

    initializeBooks(books, MAX_BOOKS);
    initializeUsers(users, MAX_USERS);

    // Simulate borrowing and returning books
    borrowBook(books, MAX_BOOKS, &users[0], 2); // Alice borrows book with ID 2
    borrowBook(books, MAX_BOOKS, &users[1], 3); // Bob borrows book with ID 3
    borrowBook(books, MAX_BOOKS, &users[0], 1); // Alice borrows book with ID 1
    returnBook(books, MAX_BOOKS, &users[0], 2); // Alice returns book with ID 2

    printLibraryStatus(books, MAX_BOOKS, users, MAX_USERS);

    return 0;
}

// Function to initialize books
void initializeBooks(Book books[], int count) {
    for (int i = 0; i < count; i++) {
        books[i].id = i + 1;
        sprintf(books[i].title, "Book Title %d", i + 1);
        sprintf(books[i].author, "Author %d", i + 1);
        books[i].isBorrowed = 0;
    }
}

// Function to initialize users
void initializeUsers(User users[], int count) {
    for (int i = 0; i < count; i++) {
        users[i].id = i + 1;
        sprintf(users[i].name, "User %d", i + 1);
        users[i].borrowedCount = 0;
        for (int j = 0; j < MAX_BOOKS; j++) {
            users[i].borrowedBooks[j] = 0;
        }
    }
}

// Function for borrowing a book
void borrowBook(Book books[], int bookCount, User *user, int bookId) {
    if (bookId < 1 || bookId > bookCount) {
        printf("%s attempted to borrow an invalid book ID %d\n", user->name, bookId);
        return;
    }
    Book *book = &books[bookId - 1];
    if (book->isBorrowed) {
        printf("%s attempted to borrow '%s' but it is already borrowed\n", user->name, book->title);
        return;
    }
    book->isBorrowed = 1;
    user->borrowedBooks[user->borrowedCount++] = bookId;
    printf("%s successfully borrowed '%s'\n", user->name, book->title);
}

// Function for returning a book
void returnBook(Book books[], int bookCount, User *user, int bookId) {
    if (bookId < 1 || bookId > bookCount) {
        printf("%s attempted to return an invalid book ID %d\n", user->name, bookId);
        return;
    }
    Book *book = &books[bookId - 1];
    if (!book->isBorrowed) {
        printf("%s attempted to return '%s' which is not borrowed\n", user->name, book->title);
        return;
    }
    book->isBorrowed = 0;

    // Remove book from user's borrowed list
    for (int i = 0; i < user->borrowedCount; i++) {
        if (user->borrowedBooks[i] == bookId) {
            for (int j = i; j < user->borrowedCount - 1; j++) {
                user->borrowedBooks[j] = user->borrowedBooks[j + 1];
            }
            user->borrowedBooks[--user->borrowedCount] = 0;
            break;
        }
    }
    printf("%s successfully returned '%s'\n", user->name, book->title);
}

// Function to print the library status
void printLibraryStatus(const Book books[], int bookCount, const User users[], int userCount) {
    printf("\nLibrary Status:\n");
    printf("Books:\n");
    for (int i = 0; i < bookCount; i++) {
        printf("  ID: %d, Title: %s, Author: %s, Status: %s\n",
               books[i].id, books[i].title, books[i].author,
               books[i].isBorrowed ? "Borrowed" : "Available");
    }

    printf("\nUsers:\n");
    for (int i = 0; i < userCount; i++) {
        printf("  User ID: %d, Name: %s, Borrowed Books: ", users[i].id, users[i].name);
        if (users[i].borrowedCount == 0) {
            printf("None");
        } else {
            for (int j = 0; j < users[i].borrowedCount; j++) {
                printf("%d ", users[i].borrowedBooks[j]);
            }
        }
        printf("\n");
    }
}
