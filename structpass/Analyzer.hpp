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


// #define HOTTEST_LOOP_CNT 10 // only the 10 hottest loops are considered 
#define ANALYSIS_FNAME "field_loop_analysis.json"
#define GROUPING_FNAME "groupings.json"



// #ifndef HEADER_ANALYZER_H 
// #define HEADER_ANALYZER_H 
#pragma once 
#include "Transformer.hpp"
#include "Types.hpp"

using namespace llvm;
using namespace std;
using namespace nlohmann;    

class Analyzer {
    public: 
        // Instance variables
        Module& M;  // Keep them as references
        ModuleAnalysisManager& MAM; 
        LLVMContext &context; 
        IRBuilder<> Builder;
        // struct table
        unordered_map<string, StructDef>    structTable;
        vector<Loop*>                       hottestLoops; 
        unordered_map<Loop*, int>           loopCntTable; 
        unordered_map<BasicBlock*, int>           bbCntTable;
        std::vector<BasicBlock*> hottestBBs; 
        StructAccessTable structAccessTable; 
        StructBBAccessTable structBBAccessTable; 
        
        StructAccessVectorTable structAccessVectorTable; 
        StructGroupingTable structGroupingTable; 
        int L, mode; 
        unordered_map<StructType*, vector<BasicBlock*>> structHottestBBTable; 
        // Transformer transformer; 

        // Constructor that uses an initialization list
        Analyzer(Module &M, ModuleAnalysisManager &MAM, int L, int mode) 
            : M(M), MAM(MAM), context(M.getContext()), Builder(IRBuilder<>(context)), L(L), mode(mode)  {   // Initialize references in the initialization list
            // this->context = M.getContext(); 
            this->structTable = getAllStructs(); 
            printStructTable(this->structTable); 
            this->loopCntTable = getLoopCntTable();

            if(mode==MODE_LOOP){ 
                this->hottestLoops = getHottestLoops(); 
                this->structAccessTable = getStructAccessTable(); 
                printStructAccessTable(this->structAccessTable);
                this->structAccessVectorTable = convertStructAccessTableToVectors(this->structAccessTable); 
                printStructAccessVectorTable(this->structAccessVectorTable);  
                saveSerializedVectorTable(this->structAccessVectorTable, ANALYSIS_FNAME); 
            }else{
                this->hottestBBs = getHottestBBs(); 
                getStructHottestBBTable(); 
                this->structBBAccessTable = getStructBBAccessTable(); 
                this->structAccessVectorTable = convertStructBBAccessTableToVectors(this->structBBAccessTable); 
                saveSerializedVectorTable(this->structAccessVectorTable, ANALYSIS_FNAME);  
            } 
        } 
        void transform(){
            unordered_set<StructDef*> needTransformation; 
            for(auto& item:this->structTable){
                if(this->structGroupingTable.find(item.first)!=this->structGroupingTable.end()) {
                    needTransformation.insert(&(item.second)); 
                }
            }
            unordered_set<StructDef*> transformedStructs; 
            for(auto &sdef: needTransformation){
                // Only "top level" structs are splitted 
                // Splitting non "top level"(i.e. is a member of another struct) will require too many levels of indirection, making the optimization not worth it.  
                // E.g. struct B contains struct A, both structs are split 
                // Accessing field a on struct requires: 
                // 1. GEP struct B substruct ptr 
                // 2. Load B substruct ptr 
                // 3. GEP struct A field ptr 
                // 4. Load struct A data 
                // It also makes the optimization exponentially harder by introducing 
                // - Inter-struct dependency graph construction: We need start by transforming structs with 0 members and recursively transform structs whose members are "ready" by replacing their original member type with new parent type
                // - Allocation + Store of substructs recursively: Whenever we transform a struct type that contains a transformed type, we must also allocate its substructs and store substruct pointers into the parent struct. If the transformed type then contains another transformed type, then we need to find the substruct from the transformed type and do alloca and store again. The overhead/complexity is insane here.  
                if(sdef->isMember.empty()){ 
                    errs() << "\n------------------------TRANSFORMING STRUCT TYPE: " << sdef->name<<"---------------------------------------------------------\n";
                    Transformer transformer(*sdef, context); 
                    vector<int> grouping = this->structGroupingTable[sdef->name]; 
                    bool all_zeros = std::all_of(grouping.begin(), grouping.end(), [](int i) { return i == 0; });
                    if(!all_zeros) 
                        transformer.transform(this->structGroupingTable[sdef->name]); 
                    errs() << "\n------------------------FINISHED TRANSFORMING STRUCT TYPE: " << sdef->name<<"---------------------------------------------------------\n";
                    transformedStructs.insert(sdef); 
                }
            }
            // not using this for now 
                // while(!(needTransformation.empty())){
                //     unordered_set<StructDef*> needErase; 
                //     for(auto &sdef: needTransformation){
                //         if(sdef->isMember.empty()){ // only transform top level for simplicity 
                //             errs() << "\n------------------------TRANSFORMING STRUCT TYPE: " << sdef->name<<"---------------------------------------------------------\n";
                //             Transformer transformer(*sdef, context); 
                //             transformer.transform(this->structGroupingTable[sdef->name]); 
                //             errs() << "\n------------------------FINISHED TRANSFORMING STRUCT TYPE: " << sdef->name<<"---------------------------------------------------------\n";
                //             errs() << "Erase struct " << sdef->st->getStructName().str() << " from set \n"; 
                //             needErase.insert(sdef); 
                //             StructType* parentStructType = transformer.parentStructType; 
                //             for(auto &sdefRemain: needTransformation){
                //                 if(sdefRemain!=sdef)  sdefRemain->replaceMemberWithType(sdef->st, parentStructType); 
                //             } 
                //         }
                //     }
                //     for(auto &sd:needErase) needTransformation.erase(sd); 
                //     break; 
                //     if(needErase.empty()) break; 
                // }
            errs() << "\n--------------------------------------------TRANSFORM PASS FINISHED-----------------------------------------------------\n";
            errs() << "\nTransformed structs: \n"; 
            for(auto &sdef: transformedStructs){
                sdef->print(errs()); 
                errs() << "\n"; 
            }
            return; 
        }
        StructGroupingTable loadStructGroupingTable(string fname){
            nlohmann::json jsonData; 
            ifstream inFile(fname); 
            inFile >> jsonData; 
            StructGroupingTable res; 
            for(const auto& [key, value] : jsonData.items()) {
                res[key] = value.get<vector<int>>(); 
            }
            this->structGroupingTable = res; 
            return res; 
        }
        void saveSerializedVectorTable(StructAccessVectorTable table, string fname ){
            errs() << "Saving serialized vector table to file: "<<fname<<"\n"; 
            ofstream outFile(fname); 
            outFile << serializeVectorTable(table).dump(4); 
            outFile.close(); 
        }
        nlohmann::json serializeVectorTable( StructAccessVectorTable table ){
            nlohmann::json jsonData; 
            jsonData = table;
            return jsonData; 
        }
        void printStructGroupingTable(StructGroupingTable table){
            for(auto& item:table){
                errs() << "Struct Name: "<< item.first << " Member Count: "<<item.second.size() << "\n"; 
                errs() << "Groupings: \t"; 
                for(int i = 0;i<item.second.size();i++){
                    errs() << i <<":" << item.second[i] << "\t"; 
                }
                errs() << "\n"; 
            }
        }
        void printStructTable( unordered_map<string, StructDef> table ){
            for(auto &item:table){
                errs()<<item.first<<"\n"; 
            }
        }
        void printStructAccessTable(StructAccessTable table){
            errs() << "\n\nPrinting Struct Access Table:\n"; 
            for(auto &item:table){
                errs() << "\n-----------------------------------------------------\n"; 
                item.second.print(errs()); 
                errs() << "-----------------------------------------------------\n";
            }
        }
        void printStructAccessVectorTable(StructAccessVectorTable table){
            for(auto &item: table){
                errs() << "Struct Name: "<<item.first<<"\n"; 
                for(int i = 0;i<item.second.size();i++){
                    errs() << "Loop "<<i<<": \t"; 
                    for(int j = 0;j<item.second[0].size();j++){
                        errs() << item.second[i][j] << "\t"; 
                    }
                    errs() << "\n"; 
                }
            }
            errs() << "\n"; 
        }
        StructAccessVectorTable convertStructAccessTableToVectors(StructAccessTable table){
            StructAccessVectorTable res;
            errs() <<" number of structs: "<<this->structTable.size() << "\n"; 
            // initialize the table for each struct 
            for(auto &item:structTable){
                string structName = item.first; 
                errs() << "Intializing struct "<<structName<<" with matrix of "<<this->L<<" x "<<item.second.st->getNumElements()<<"\n"; 
                res[structName] = vector<vector<int>>(this->L, vector<int>(item.second.st->getNumElements(),0)); 
                // init cnt to 0 
            }
            // errs() << "bro???"; 
            for(auto &item:table){
                auto key = item.first; 
                string structName = item.second.key.st->getStructName().str();
                // errs() << "st: "<<item.second.key.st << " name: "<<structName << "\n";  
                int loopIdx = item.second.key.loopIdx; 
                int memberIdx = item.second.key.memberIdx; 
                int accessCnt = item.second.getTotalAccessCnt(); 
                // errs() << "access cnt: "<<accessCnt<<"\n"; 
                // errs() <<" usage cnt: "<<item.second.usage_cnts.size()<< "\n"; 
                res[structName][loopIdx][memberIdx] = accessCnt; 
            }
            // errs() << "huh???"; 
            return res; 
        }
        StructAccessVectorTable convertStructBBAccessTableToVectors(StructBBAccessTable table){
            StructAccessVectorTable res;
            errs() <<" number of structs: "<<this->structTable.size() << "\n"; 
            // initialize the table for each struct 
            for(auto &item:structTable){
                string structName = item.first; 
                errs() << "Intializing struct "<<structName<<" with matrix of "<<this->L<<" x "<<item.second.st->getNumElements()<<"\n"; 
                res[structName] = vector<vector<int>>(this->L, vector<int>(item.second.st->getNumElements(),0)); 
                // init cnt to 0 
            }
            // errs() << "bro???"; 
            for(auto &item:table){
                auto key = item.first; 
                string structName = item.second.key.st->getStructName().str();
                // errs() << "st: "<<item.second.key.st << " name: "<<structName << "\n";  
                int bbIdx = item.second.key.bbIdx; 
                int memberIdx = item.second.key.memberIdx; 
                int accessCnt = item.second.getTotalAccessCnt(); 
                // errs() << "access cnt: "<<accessCnt<<"\n"; 
                // errs() <<" usage cnt: "<<item.second.usage_cnts.size()<< "\n"; 
                res[structName][bbIdx][memberIdx] = accessCnt; 
            }
            // errs() << "huh???"; 
            return res; 
        }

    private: 
        void  getStructHottestBBTable(){
            for(auto& item:this->structTable){
                string name = item.first; 
                StructDef sdef = item.second; 
                // Step 1: Copy entries to a vector
                std::vector<std::pair<BasicBlock*, int>> bbVector; 
                for(BasicBlock* bb:sdef.gepAccessBBs){
                    bbVector.push_back({bb, bbCntTable[bb]}); 
                }

                // Step 2: Sort by values in descending order
                std::sort(bbVector.begin(), bbVector.end(),
                        [](const std::pair<BasicBlock*, int> &a, const std::pair<BasicBlock*, int> &b) {
                            return a.second > b.second; // Compare by value
                        });

                // Step 3: Extract the top N BasicBlocks
                std::vector<BasicBlock*> hottestBBs(this->L, nullptr); 
                for (int i = 0; i < std::min(this->L, (int)bbVector.size()); ++i) {
                    BasicBlock* bb = bbVector[i].first;  
                    errs() << "\nHottest BB "<< i<< "(" << bbVector[i].second << "): \n"; 
                    bb->print(errs()); 
                    hottestBBs[i]=bb;
                }
                sdef.hottestGepAccessBBs = hottestBBs; 
                this->structTable[name] = sdef; 
            }
        }
        // this gets the table of struct_name:structDef 
        unordered_map<string, StructDef> getAllStructs(){
            errs()<<"All structs: \n"; 
            auto structs =  this->M.getIdentifiedStructTypes(); 
            // construct isMember/hasMember table 
            map<StructType*, unordered_set<StructType*>> isMemberMap; 
            map<StructType*, unordered_set<StructType*>> hasMemberMap; 
            for(auto &s:structs){
                if(StructType* st = dyn_cast<StructType>(s)){
                    for(int i = 0;i<s->getNumElements();i++){
                        auto member = st->getElementType(i); 
                        // if(member->isStructTy()){
                        StructType* memberSt = isStructType(member); 
                        if(memberSt!=nullptr){
                            // errs() << "Member: " << member->getStructName().str() << " is a struct!!!\n"; 
                            // StructType* memberStruct = dyn_cast<StructType>(member);
                            StructType* memberStruct = memberSt;  
                            isMemberMap[memberStruct].insert(st); 
                            hasMemberMap[st].insert(memberStruct); 
                        }
                    }
                }
            }

            unordered_map<string, StructDef> struct_table;
            unordered_set<StructType*> struct_type_set; 
            for(auto &s: structs){
                StructType* st = dyn_cast<StructType>(s); 
                if(st!=nullptr && (st->getStructName().str().find("struct")!=string::npos)){
                    StructDef st_def; 
                    st_def.name = s->getName(); 
                    st_def.st = st; 
                    for(int i = 0;i<s->getNumElements();i++) {
                        auto member = st->getElementType(i); 
                        st_def.members.push_back(member); 
                    }
                    st_def.isMember = isMemberMap[st]; 
                    st_def.hasMember = hasMemberMap[st]; 
                    st_def.print(errs()); 
                    struct_table[st->getStructName().str()] = st_def; 
                    struct_type_set.insert(st); 
                }
            } 
            // get all usages and alloca of structs here. 
            for(auto &F:M){
                for(auto &BB: F){
                    for(auto &I:BB){
                        StructType* structType; 
                        structType = isStructAllocaUsage(&I, struct_type_set); 
                        if(structType){
                            string structName = structType->getStructName().str(); 
                            errs() << "Alloca of struct: "<< structType->getStructName().str() << "\n"; 
                            structType->print(errs()); 
                            errs() << "\nInst: "; 
                            I.print(errs()); 
                            errs() << "\n\n"; 
                            struct_table[structName].allocaUsages.insert(dyn_cast<AllocaInst>(&I)); 
                            continue; 
                        }
                        structType = isStructGEPUsage(&I, struct_type_set); 
                        if(structType){
                            string structName = structType->getStructName().str(); 
                            errs() << "GEP of struct: "<< structType->getStructName().str() << "\n"; 
                            structType->print(errs()); 
                            errs() << "\nInst: "; 
                            I.print(errs()); 
                            errs() << "\n\n";
                            BasicBlock* bb = I.getParent(); 
                            if(bb) struct_table[structName].gepAccessBBs.insert(bb); 
                            struct_table[structName].gepUsages.insert(dyn_cast<GetElementPtrInst>(&I)); 
                            continue; 
                        }
                    }
                }
            }
            

            return struct_table; 
        }

        StructType* isStructType(Type* t) {
            if (!t) {
                return nullptr; // Handle null input
            }

            // Direct case: Check if the type is a StructType
            if (auto* structTy = dyn_cast<StructType>(t)) {
                return structTy;
            }

            if (auto *arrTy = dyn_cast<ArrayType>(t)){
                auto *structType = dyn_cast<StructType>(arrTy->getElementType()); 
                return structType; 
            }

            return nullptr; // Not a struct or related type
        }
        StructType* isStructAllocaUsage(Instruction* inst, unordered_set<StructType*> struct_type_set){
            if(AllocaInst* allocaInst = dyn_cast<AllocaInst>(inst)){
                if(allocaInst->getAllocatedType()->isArrayTy()){
                    ArrayType *arrayType = cast<ArrayType>(allocaInst->getAllocatedType());
                    if (arrayType->getElementType()->isStructTy()){
                        StructType* structType = dyn_cast<StructType>(arrayType->getElementType());
                        if(struct_type_set.find(structType)!=struct_type_set.end()) 
                            return structType;  
                        return nullptr;   
                    }
                }
                // a struct instantiation 
                if(allocaInst->getAllocatedType()->isStructTy()){
                    auto structType =  dyn_cast<StructType>(allocaInst->getAllocatedType()); 
                    if(struct_type_set.find(structType)!=struct_type_set.end()) 
                        return structType;
                    return nullptr; 
                }
            }
            return nullptr; 
        }
        StructType* isStructGEPUsage(Instruction* inst, unordered_set<StructType*> struct_type_set){
            if(GetElementPtrInst* gepInst = dyn_cast<GetElementPtrInst>(inst)){
                if(gepInst->getSourceElementType()->isArrayTy()){
                    ArrayType* arrayType = cast<ArrayType>(gepInst->getSourceElementType()); 
                    if(arrayType->getElementType()->isStructTy()){
                        StructType* structType = dyn_cast<StructType>(arrayType->getElementType()); 
                        if(struct_type_set.find(structType)!=struct_type_set.end()) 
                            return structType;
                        return nullptr; 
                    }
                }
                if(gepInst->getSourceElementType()->isStructTy()){
                    // this is a potential struct access 
                    StructType *structType = cast<StructType>(gepInst->getSourceElementType());
                    if(struct_type_set.find(structType)!=struct_type_set.end()) 
                        return structType;
                    return nullptr; 
                } 
            } 
            return nullptr; 
        }
        vector<BasicBlock*> getHottestBBs(){
            // Step 1: Copy entries to a vector
            std::vector<std::pair<BasicBlock*, int>> bbVector(bbCntTable.begin(), bbCntTable.end());

            // Step 2: Sort by values in descending order
            std::sort(bbVector.begin(), bbVector.end(),
                    [](const std::pair<BasicBlock*, int> &a, const std::pair<BasicBlock*, int> &b) {
                        return a.second > b.second; // Compare by value
                    });

            // Step 3: Extract the top N BasicBlocks
            std::vector<BasicBlock*> hottestBBs(this->L, nullptr); 
            for (int i = 0; i < std::min(this->L, (int)bbVector.size()); ++i) {
                BasicBlock* bb = bbVector[i].first;  
                errs() << "\nHottest BB "<< i<< "(" << bbVector[i].second << "): \n"; 
                bb->print(errs()); 
                hottestBBs[i]=bb;
            }

            return hottestBBs;
        }
        vector<Loop*> getHottestLoops() {
            std::unordered_map<Loop*, int> loop_cnt_table = getLoopCntTable();
            // std::unordered_map<Loop*, int> res;
            vector<Loop*> res; 
            std::vector<std::pair<Loop*, int>> loops;

            // Copy all entries from the map to the vector
            for (const auto &item : loop_cnt_table) {
                loops.push_back(item);
            }

            // Sort the loops based on the count (in ascending order)
            std::sort(loops.begin(), loops.end(), LoopCntComparator());

            // Output sorted loops and optionally fill `res` with some logic
            for (const auto &l : loops) {
                errs() << "Hot Loop Cnt: " << l.second << "\n";
                // Optionally insert into result map if needed
                // res.insert(l);  // Copy pair (Loop*, count) into the result map
                res.push_back(l.first); 
                if(res.size() >= this->L) break; 
            }

            return res;
        }
        unordered_map<Loop*, int> getLoopCntTable() {
            /*
                hottest loop will be loops with the most execution counts
            */
            unordered_map<Loop*, int> loop_cnt_table; 
            for (Function &F : M) {
                FunctionAnalysisManager &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
                if (F.isDeclaration()) continue; // Only consider definitions

                // Get loop analysis and block frequency analysis
                LoopAnalysis::Result &LI = FAM.getResult<LoopAnalysis>(F);
                BlockFrequencyAnalysis::Result &BFA = FAM.getResult<BlockFrequencyAnalysis>(F);

                // Iterate over loops
                int loop_id = 0; 
                for (Loop *L : LI) {
                    errs() << "Loop: \n"; 
                    L->print(errs()); // For debugging, print the loop information
                    int loop_execution_cnt = 0; 
                    // Iterate over basic blocks inside the loop
                    for (BasicBlock *BB : L->blocks()) {
                        errs() << "\nBB "<<BB<<": "; 
                        BB->print(errs()); 
                        errs() << "\n"; 
                        if(auto profCnt = BFA.getBlockProfileCount(BB)) {
                            auto blockExecCount = profCnt.value(); 
                            bbCntTable[BB] = blockExecCount;
                            errs() << "Block execution count: " << blockExecCount << "\n";
                            loop_execution_cnt += blockExecCount;
                        }
                        // auto blockExecCount = BFA.getBlockProfileCount(BB).value(); 
                        // bbCntTable[BB] = blockExecCount;
                        // errs() << "Block execution count: " << blockExecCount << "\n";
                        // loop_execution_cnt += blockExecCount; 
                    }

                    errs()<<"loop dynamic inst count: "<<loop_execution_cnt<<"\n"; 
                    loop_id ++; 
                    loop_cnt_table[L] = loop_execution_cnt; 
                }
            }

            return loop_cnt_table; 
        }
        StructBBAccessTable getStructBBAccessTable(){
            StructBBAccessTable struct_bb_access_table; 
            for(auto &item:this->structTable){
                StructDef sdef = item.second; 
                vector<llvm::BasicBlock*> hottestBBs = sdef.hottestGepAccessBBs; 
                for(int i = 0;i<hottestBBs.size();i++){
                    BasicBlock* bb = hottestBBs[i]; 
                    if(bb){
                        for(Instruction& inst:(*bb)){
                            // find accesses of the members 
                            if(GetElementPtrInst* gepInst = dyn_cast<GetElementPtrInst>(&inst)){
                                    if(gepInst->getSourceElementType()->isStructTy()){
                                    // this is a potential struct access 
                                    StructType *st = cast<StructType>(gepInst->getSourceElementType()); 
                                    if(st && st->getStructName().str().find("struct")!=string::npos){
                                        errs() << "Potential struct access to struct name: "<<st->getStructName()<<"in loop "<<i<<"\n";
                                        gepInst->print(errs()); 
                                        errs() << "\n"; 
                                        Value *memberIdxOperand = gepInst->getOperand(gepInst->getNumOperands()-1); // last operand is 
                                        ConstantInt *memberIdxInt = dyn_cast<ConstantInt>(memberIdxOperand); 
                                        if(memberIdxInt) {
                                            int memberIdx = memberIdxInt->getSExtValue(); 
                                            errs() << "Accessing member idx: "<< memberIdx <<"\n"; 
                                            errs() << "Users: "<<"\n"; 
                                            StructMemberBBAccessKey memberAccessKey = { st, memberIdx, bb, i };
                                            struct_bb_access_table[memberAccessKey].key = memberAccessKey; 
                                            for(llvm::User* user:gepInst->users()) {
                                                errs() << user << "\n"; 
                                                user->print(errs()); 
                                                errs() << "\ncount: "<<bbCntTable[bb]<<"\n"; 
                                                errs() << "\n"; 
                                                int userExecCnt = bbCntTable[bb]; 
                                                if(Instruction* userInst = dyn_cast<Instruction>(user)) {
                                                    struct_bb_access_table[memberAccessKey].usages.insert(userInst); 
                                                    struct_bb_access_table[memberAccessKey].usage_cnts[userInst] = userExecCnt; 
                                                }
                                            }
                                        }
                                    }
                                    }
                            }
                        }
                    }
                }
            }
            return struct_bb_access_table; 
        }
        StructAccessTable getStructAccessTable(){
            StructAccessTable struct_access_table; 
            for(int i = 0;i<hottestLoops.size();i++) {
                Loop* loop = hottestLoops[i]; 
                for(BasicBlock* bb:loop->getBlocks()){
                    for(Instruction& inst:(*bb)){
                        // find accesses of the members 
                        if(GetElementPtrInst* gepInst = dyn_cast<GetElementPtrInst>(&inst)){
                             if(gepInst->getSourceElementType()->isStructTy()){
                                // this is a potential struct access 
                                StructType *st = cast<StructType>(gepInst->getSourceElementType()); 
                                if(st && st->getStructName().str().find("struct")!=string::npos){
                                    errs() << "Potential struct access to struct name: "<<st->getStructName()<<"in loop "<<i<<"\n";
                                    gepInst->print(errs()); 
                                    errs() << "\n"; 
                                    Value *memberIdxOperand = gepInst->getOperand(gepInst->getNumOperands()-1); // last operand is 
                                    ConstantInt *memberIdxInt = dyn_cast<ConstantInt>(memberIdxOperand); 
                                    if(memberIdxInt) {
                                        int memberIdx = memberIdxInt->getSExtValue(); 
                                        errs() << "Accessing member idx: "<< memberIdx <<"\n"; 
                                        errs() << "Users: "<<"\n"; 
                                        StructMemberAccessKey memberAccessKey = { st, memberIdx, loop, i };
                                        struct_access_table[memberAccessKey].key = memberAccessKey; 
                                        for(llvm::User* user:gepInst->users()) {
                                            errs() << user << "\n"; 
                                            user->print(errs()); 
                                            errs() << "\ncount: "<<bbCntTable[bb]<<"\n"; 
                                            errs() << "\n"; 
                                            int userExecCnt = bbCntTable[bb]; 
                                            if(Instruction* userInst = dyn_cast<Instruction>(user)) {
                                                struct_access_table[memberAccessKey].usages.insert(userInst); 
                                                struct_access_table[memberAccessKey].usage_cnts[userInst] = userExecCnt; 
                                            }
                                        }
                                    }
                                }
                             }
                        }
                    }
                } 
            }
            return struct_access_table; 
        }
}; 
// #endif // !