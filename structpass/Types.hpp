////===-------------------------------------------------------------------===//
#include "llvm/Pass.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"


/* *******Implementation Starts Here******* */
// You can include more Header files here
#include <bits/stdc++.h> 
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopInfo.h"
// #include "llvm/Analysis/BlockFrequencyAnalysis.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream> 
#include <fstream> 
#include "json.hpp" 
using namespace llvm;
using namespace std;
using namespace nlohmann;    
#pragma once 
struct StructMemberAccessKey{
    StructType* st; 
    int memberIdx; 
    Loop* loop; 
    int loopIdx; 
    // for hashing 
    bool operator==(const StructMemberAccessKey& other) const {
        return st == other.st && memberIdx == other.memberIdx &&
               loop == other.loop && loopIdx == other.loopIdx;
    }
    void print(raw_ostream &O){
        if (st) O << "Struct Name:\t" << st->getStructName() << "\t";
        O << "Member Idx:\t" << memberIdx << "\t";
        O << "Loop Idx:\t" << loopIdx << "\n";
        O<<"\n"; 
    } 
}; 
struct StructMemberAccessRecord{
    StructMemberAccessKey key; 
    unordered_set<llvm::Instruction*> usages; 
    unordered_map<llvm::Instruction*, int> usage_cnts;
    // for hashing 
    bool operator==(const StructMemberAccessRecord& other) const {
        return (key==other.key) && (usages==other.usages) && (usage_cnts == other.usage_cnts); 
    }
    int getTotalAccessCnt(){
        int sum = 0; 
        for(auto &item:usage_cnts){
            sum += item.second; 
        }
        return sum; 
    }
    void print(raw_ostream &O){
        O << "\tKey:\t";
        key.print(O); 
        O << "\n"; 

        for (auto &usage : usage_cnts) {
            errs() << "\tUsage:\t"; 
            usage.first->print(errs()); 
            errs() << "\tTotal Access Count:\t" << usage.second << "\n"; 
        }
        errs() << "\nTotal Count: "<<getTotalAccessCnt()<<"\n"; 
    }
}; 

// Custom hash function for StructMemberAccessKey
struct StructMemberAccessKeyHasher {
    std::size_t operator()(const StructMemberAccessKey& key) const {
        std::size_t h1 = std::hash<StructType*>()(key.st);
        std::size_t h2 = std::hash<int>()(key.memberIdx);
        std::size_t h3 = std::hash<Loop*>()(key.loop);
        std::size_t h4 = std::hash<int>()(key.loopIdx);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);  // Combine the hash values
    }
};

// Custom hash function for StructMemberAccessRecord
struct StructMemberAccessRecordHasher {
    std::size_t operator()(const StructMemberAccessRecord& record) const {
        StructMemberAccessKeyHasher keyHasher;
        std::size_t keyHash = keyHasher(record.key);
        std::size_t usagesHash = std::hash<std::size_t>()(record.usages.size());  // Example, could be more complex
        std::size_t usageCntsHash = std::hash<std::size_t>()(record.usage_cnts.size());
        return keyHash ^ (usagesHash << 1) ^ (usageCntsHash << 2);
    }
};

typedef unordered_map<StructMemberAccessKey, StructMemberAccessRecord, StructMemberAccessKeyHasher> StructAccessTable; 
typedef unordered_map<string, vector<vector<int>>> StructAccessVectorTable; 
// Matrix[i][j]=loop i, member idx j  
typedef unordered_map<string, vector<int>> StructGroupingTable; 


typedef map<int, vector<llvm::Type*>> SubstructMembersTable; // group number : members 
typedef map<string, SubstructMembersTable> StructSplittingTable;
typedef unordered_map<int, StructType*> StructGroupIdxTable; 

// < <originalStructType, memberIdx>, <substructType, substructMemberIdx> >
typedef map<pair<Type*, int>, pair<Type*, int>> StructMemberMappingTable;  
struct StructInstanceMapping{
    Value* originalStructInstance; 
    Value* parentStructInstance; 
    unordered_map<Type*, Value*> substructInstances; // maps from sub-struct to the desired instance 
};


// Function to print the StructMemberMappingTable to errs()
void printStructMemberMappingTable(const StructMemberMappingTable& mappingTable) {
    // Iterate over each entry in the map
    for (const auto& entry : mappingTable) {
        // Extract the key and value from the map entry
        const auto& key = entry.first;  // key: pair<Type*, int>
        const auto& value = entry.second; // value: pair<Type*, int>

        // Print the key: originalStructType and memberIdx
        errs() << "Original Struct Type: ";
        if (key.first) {
            key.first->print(errs()); // Print Type* (originalStructType)
        } else {
            errs() << "null";
        }
        errs() << ", Member Index: " << key.second << "\n";

        // Print the value: substructType and substructMemberIdx
        errs() << "Substruct Type: ";
        if (value.first) {
            value.first->print(errs()); // Print Type* (substructType)
        } else {
            errs() << "null";
        }
        errs() << ", Substruct Member Index: " << value.second << "\n";
    }
}

// Function to print the StructInstanceMapping to errs()
void printStructInstanceMapping(const StructInstanceMapping& mapping) {
    // Print the originalStructInstance
    errs() << "Original Struct Instance: ";
    if (mapping.originalStructInstance) {
        mapping.originalStructInstance->print(errs());
    } else {
        errs() << "null";
    }
    errs() << "\n";

    // Print the parentStructInstance
    errs() << "Parent Struct Instance: ";
    if (mapping.parentStructInstance) {
        mapping.parentStructInstance->print(errs());
    } else {
        errs() << "null";
    }
    errs() << "\n";

    // Print the substructInstances map
    errs() << "Substruct Instances: \n";
    for (const auto& entry : mapping.substructInstances) {
        errs() << "Substruct Type: ";
        if (entry.first) {
            entry.first->print(errs()); // print Type*
        } else {
            errs() << "null";
        }
        errs() << ", Instance: ";
        if (entry.second) {
            entry.second->print(errs()); // print Value*
        } else {
            errs() << "null";
        }
        errs() << "\n";
    }
}


typedef map<Value*, StructInstanceMapping> StructInstanceMappingTable;  

struct StructSplitResult{
    StructType* originalStruct; 
    StructType* parentStruct;
    unordered_map<int, StructType*> groupIdxToSubstruct; 
    vector<Type*> substructTypes; 
    vector<Type*> substructPointers;  
    unordered_map<Type*, int> substructPtrToIdx; 
    vector<int> groupings; 
}; 

struct StructDef{
    string name; 
    StructType* st; 
    vector<llvm::Type*> members;
    unordered_set<llvm::AllocaInst*> allocaUsages; 
    unordered_set<llvm::GetElementPtrInst*> gepUsages;  
    unordered_set<llvm::StructType*> isMember;  // struct types that has this struct as a member 
    unordered_set<llvm::StructType*> hasMember; // 
    void print(raw_ostream &O){
        O<<"----------------Struct Def Information-------------------------\n"; 
        O<<"Struct name: "<<name<<" \n"; 
        O<<"Members: \n"; 
        for(int i = 0;i<members.size();i++){
            O<<"Member "<<i<<": "; 
            members[i]->print(O); 
            if(members[i]->isPointerTy()) O<<" Is Pointer: "<<members[i]->isPointerTy(); 
            if(members[i]->isStructTy()) O<<" Is Struct: "<<members[i]->isStructTy(); 
            O<<"\n";  
        }
        O<<"\nHas Member Struct Types: \n"; 
        for(auto &t:hasMember){
            t->print(errs());
            errs() << "\n";  
        }
        
        O<<"\nIs a Member Struct Type of: \n"; 
        for(auto &t:isMember){
            t->print(errs());
            errs() << "\n";  
        }
        O<<"\n-------------------------------------------------------------------------------------------\n"; 
    }
    void replaceMemberWithType(StructType* oldType, StructType* newType){
        errs() << "Replacing old type: " << oldType->getStructName().str() << " with  new type: " << newType->getStructName().str()<<"\n"; 
        for(int i = 0;i<members.size();i++){
            if(StructType* st = dyn_cast<StructType>(members[i])){
                if(st==oldType){
                    members[i] = newType; 
                    this->hasMember.erase(oldType); 
                }
            }
        }
    }
}; 
typedef unordered_map<string, StructDef> StructMap; 
struct LoopCntComparator {
    bool operator() (const std::pair<Loop*, int> &l1, const std::pair<Loop*, int> &l2) const {
        return l1.second > l2.second;  // Compare based on loop counts
    }
};