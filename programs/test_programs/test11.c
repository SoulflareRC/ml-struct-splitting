#include <stdio.h>
#include <string.h>

#define MAX_BOOKS 5
#define MAX_MEMBERS 3
#define MAX_BORROWED_BOOKS 3
#define MAX_NAME_LEN 50

// Struct for Book
typedef struct {
    int id;
    char title[MAX_NAME_LEN];
    char author[MAX_NAME_LEN];
    int isAvailable; // 1 if available, 0 if borrowed
} Book;

// Struct for Member
typedef struct {
    int id;
    char name[MAX_NAME_LEN];
    int borrowedBooks[MAX_BORROWED_BOOKS]; // Store book IDs
    int borrowCount; // Number of books currently borrowed
} Member;

// Struct for Library
typedef struct {
    Book books[MAX_BOOKS];
    Member members[MAX_MEMBERS];
} Library;

// Function prototypes
void initializeLibrary(Library* library);
void borrowBook(Library* library, int memberId, int bookId);
void returnBook(Library* library, int memberId, int bookId);
void printLibraryStatus(const Library* library);

int main() {
    Library library;
    initializeLibrary(&library);

    // Simulate borrowing and returning books
    borrowBook(&library, 1, 101);
    borrowBook(&library, 1, 102);
    borrowBook(&library, 2, 103);
    returnBook(&library, 1, 101);
    borrowBook(&library, 3, 104);

    // Print final library status
    printLibraryStatus(&library);

    return 0;
}

// Function to initialize library with sample data
void initializeLibrary(Library* library) {
    // Initialize books
    library->books[0] = (Book){101, "1984", "George Orwell", 1};
    library->books[1] = (Book){102, "To Kill a Mockingbird", "Harper Lee", 1};
    library->books[2] = (Book){103, "The Great Gatsby", "F. Scott Fitzgerald", 1};
    library->books[3] = (Book){104, "Moby-Dick", "Herman Melville", 1};
    library->books[4] = (Book){105, "Pride and Prejudice", "Jane Austen", 1};

    // Initialize members
    library->members[0] = (Member){1, "Alice", {0}, 0};
    library->members[1] = (Member){2, "Bob", {0}, 0};
    library->members[2] = (Member){3, "Charlie", {0}, 0};
}

// Function for a member to borrow a book
void borrowBook(Library* library, int memberId, int bookId) {
    for (int i = 0; i < MAX_MEMBERS; i++) {
        if (library->members[i].id == memberId) {
            Member* member = &library->members[i];

            // Check if the member can borrow more books
            if (member->borrowCount >= MAX_BORROWED_BOOKS) {
                printf("Member %s cannot borrow more books (limit reached).\n", member->name);
                return;
            }

            // Find the book
            for (int j = 0; j < MAX_BOOKS; j++) {
                if (library->books[j].id == bookId) {
                    Book* book = &library->books[j];

                    // Check if the book is available
                    if (book->isAvailable) {
                        book->isAvailable = 0;
                        member->borrowedBooks[member->borrowCount++] = bookId;
                        printf("Book '%s' borrowed by %s.\n", book->title, member->name);
                        return;
                    } else {
                        printf("Book '%s' is currently unavailable.\n", book->title);
                        return;
                    }
                }
            }

            printf("Book with ID %d not found.\n", bookId);
            return;
        }
    }

    printf("Member with ID %d not found.\n", memberId);
}

// Function for a member to return a book
void returnBook(Library* library, int memberId, int bookId) {
    for (int i = 0; i < MAX_MEMBERS; i++) {
        if (library->members[i].id == memberId) {
            Member* member = &library->members[i];

            // Find the book in the member's borrowed list
            for (int j = 0; j < member->borrowCount; j++) {
                if (member->borrowedBooks[j] == bookId) {
                    // Mark book as returned
                    for (int k = 0; k < MAX_BOOKS; k++) {
                        if (library->books[k].id == bookId) {
                            library->books[k].isAvailable = 1;
                            break;
                        }
                    }

                    // Remove the book from the member's list
                    for (int k = j; k < member->borrowCount - 1; k++) {
                        member->borrowedBooks[k] = member->borrowedBooks[k + 1];
                    }
                    member->borrowedBooks[--member->borrowCount] = 0;

                    printf("Book with ID %d returned by %s.\n", bookId, member->name);
                    return;
                }
            }

            printf("Member %s has not borrowed book with ID %d.\n", member->name, bookId);
            return;
        }
    }

    printf("Member with ID %d not found.\n", memberId);
}

// Function to print the status of the library
void printLibraryStatus(const Library* library) {
    printf("\nLibrary Status:\n");
    printf("Books:\n");
    for (int i = 0; i < MAX_BOOKS; i++) {
        const Book* book = &library->books[i];
        printf("  [%d] %s by %s - %s\n",
               book->id, book->title, book->author,
               book->isAvailable ? "Available" : "Borrowed");
    }

    printf("\nMembers:\n");
    for (int i = 0; i < MAX_MEMBERS; i++) {
        const Member* member = &library->members[i];
        printf("  [%d] %s - Borrowed Books: ", member->id, member->name);
        for (int j = 0; j < member->borrowCount; j++) {
            printf("%d ", member->borrowedBooks[j]);
        }
        printf("\n");
    }
}
