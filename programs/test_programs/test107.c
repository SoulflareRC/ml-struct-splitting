#include <stdio.h>
#include <string.h>

#define MAX_ACCOUNTS 10
#define MAX_TRANSACTIONS 50

// Struct for Transaction
typedef struct {
    char type[10];  // "Deposit", "Withdrawal", "Loan Repayment"
    double amount;
    char date[11];  // Date in format: YYYY-MM-DD
} Transaction;

// Struct for Account
typedef struct {
    int accountNumber;
    char accountHolder[50];
    double balance;
    double interestRate;  // Interest rate in percentage (annual)
    Transaction transactions[MAX_TRANSACTIONS];
    int transactionCount;
} Account;

// Struct for Bank
typedef struct {
    Account accounts[MAX_ACCOUNTS];
    int accountCount;
} Bank;

// Function prototypes
void initializeBank(Bank *bank);
void createAccount(Bank *bank, int accountNumber, const char *accountHolder, double initialDeposit, double interestRate);
void deposit(Bank *bank, int accountNumber, double amount, const char *date);
void withdraw(Bank *bank, int accountNumber, double amount, const char *date);
void applyInterest(Bank *bank, int accountNumber);
void repayLoan(Bank *bank, int accountNumber, double amount, const char *date);
void printTransactionHistory(const Account *account);
void displayAccountDetails(const Account *account);

int main() {
    // Initialize Bank
    Bank bank;
    initializeBank(&bank);

    // Create accounts
    createAccount(&bank, 1001, "John Doe", 1000.0, 5.0);  // Savings account with 5% annual interest
    createAccount(&bank, 1002, "Jane Smith", 500.0, 0.0);  // Checking account (no interest)
    createAccount(&bank, 1003, "Alice Brown", 2000.0, 7.0);  // Savings account with 7% annual interest

    // Perform transactions
    deposit(&bank, 1001, 500.0, "2024-11-21");
    withdraw(&bank, 1002, 100.0, "2024-11-22");
    deposit(&bank, 1003, 1500.0, "2024-11-23");

    // Apply interest
    applyInterest(&bank, 1001);
    applyInterest(&bank, 1003);

    // Repay loans (simulate as withdrawals from an account)
    repayLoan(&bank, 1002, 50.0, "2024-11-24");

    // Display account details and transaction histories
    for (int i = 0; i < bank.accountCount; i++) {
        displayAccountDetails(&bank.accounts[i]);
        printTransactionHistory(&bank.accounts[i]);
    }

    return 0;
}

// Initialize the bank with no accounts
void initializeBank(Bank *bank) {
    bank->accountCount = 0;
}

// Create a new account with initial deposit
void createAccount(Bank *bank, int accountNumber, const char *accountHolder, double initialDeposit, double interestRate) {
    if (bank->accountCount < MAX_ACCOUNTS) {
        Account *newAccount = &bank->accounts[bank->accountCount++];
        newAccount->accountNumber = accountNumber;
        strcpy(newAccount->accountHolder, accountHolder);
        newAccount->balance = initialDeposit;
        newAccount->interestRate = interestRate;
        newAccount->transactionCount = 0;

        // Record the initial deposit transaction
        strcpy(newAccount->transactions[0].type, "Deposit");
        newAccount->transactions[0].amount = initialDeposit;
        strcpy(newAccount->transactions[0].date, "2024-11-20");
        newAccount->transactionCount = 1;
    } else {
        printf("Account creation failed. Maximum number of accounts reached.\n");
    }
}

// Deposit money into an account
void deposit(Bank *bank, int accountNumber, double amount, const char *date) {
    for (int i = 0; i < bank->accountCount; i++) {
        if (bank->accounts[i].accountNumber == accountNumber) {
            bank->accounts[i].balance += amount;

            // Record the transaction
            Transaction *trans = &bank->accounts[i].transactions[bank->accounts[i].transactionCount++];
            strcpy(trans->type, "Deposit");
            trans->amount = amount;
            strcpy(trans->date, date);

            return;
        }
    }
    printf("Account not found.\n");
}

// Withdraw money from an account
void withdraw(Bank *bank, int accountNumber, double amount, const char *date) {
    for (int i = 0; i < bank->accountCount; i++) {
        if (bank->accounts[i].accountNumber == accountNumber) {
            if (bank->accounts[i].balance >= amount) {
                bank->accounts[i].balance -= amount;

                // Record the transaction
                Transaction *trans = &bank->accounts[i].transactions[bank->accounts[i].transactionCount++];
                strcpy(trans->type, "Withdrawal");
                trans->amount = amount;
                strcpy(trans->date, date);
            } else {
                printf("Insufficient funds for withdrawal.\n");
            }
            return;
        }
    }
    printf("Account not found.\n");
}

// Apply interest to a savings account
void applyInterest(Bank *bank, int accountNumber) {
    for (int i = 0; i < bank->accountCount; i++) {
        if (bank->accounts[i].accountNumber == accountNumber) {
            if (bank->accounts[i].interestRate > 0) {
                double interest = bank->accounts[i].balance * (bank->accounts[i].interestRate / 100);
                bank->accounts[i].balance += interest;

                // Record the interest transaction
                Transaction *trans = &bank->accounts[i].transactions[bank->accounts[i].transactionCount++];
                strcpy(trans->type, "Interest");
                trans->amount = interest;
                strcpy(trans->date, "2024-12-31");
            }
            return;
        }
    }
    printf("Account not found.\n");
}

// Simulate loan repayment by withdrawing from the account
void repayLoan(Bank *bank, int accountNumber, double amount, const char *date) {
    withdraw(bank, accountNumber, amount, date);

    // Record the loan repayment transaction
    for (int i = 0; i < bank->accountCount; i++) {
        if (bank->accounts[i].accountNumber == accountNumber) {
            Transaction *trans = &bank->accounts[i].transactions[bank->accounts[i].transactionCount++];
            strcpy(trans->type, "Loan Repayment");
            trans->amount = amount;
            strcpy(trans->date, date);
            return;
        }
    }
    printf("Account not found.\n");
}

// Print transaction history for a given account
void printTransactionHistory(const Account *account) {
    printf("Transaction History for Account #%d (%s):\n", account->accountNumber, account->accountHolder);
    for (int i = 0; i < account->transactionCount; i++) {
        printf("%s: %.2f on %s\n", account->transactions[i].type, account->transactions[i].amount, account->transactions[i].date);
    }
}

// Display account details
void displayAccountDetails(const Account *account) {
    printf("Account #%d (%s)\n", account->accountNumber, account->accountHolder);
    printf("Balance: %.2f\n", account->balance);
    printf("Interest Rate: %.2f%%\n", account->interestRate);
}
