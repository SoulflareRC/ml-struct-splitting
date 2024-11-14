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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

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
 
#include "Types.hpp" 
#pragma once 

#define MAX_N 3 
using namespace std; 
using namespace llvm;  

class Transformer{
    /*
        Transformer only works on 1 original struct per time. 
        This makes things more flexible at analyzer level things simpler. 
    */
    public: 

    /*
        To replace original struct usage, there are 2 scenarios: 
        1. AllocaInst: 
            - A usage of AllocaInst = instantiation of a struct 
            - We need to instantiate 
                - The parent struct 
                - all substructs 
            - We also need to point parent struct's substruct pointers to the new substruct instances, which needs StoreInst 
                - 
            - 
    */
    LLVMContext& context; 
    StructDef structDef; 
    vector<int> groupings; 
    map<int, vector<llvm::Type*>> substructMembersTable; 
    unordered_map<Type*, int> substructToParentIdx; 
    vector<pair<int, int>> memberIdxTable; 
    StructType* originalStructType; 
    StructType* parentStructType; 
    unordered_map<int, StructType*> groupIdxToSubstruct; 

    // maps from original struct's member to the substruct's member 
    map<int, pair<Type*, int>> originalStructMemberToSubstructMember; 

    // <original struct instance, substruct type> : substruct instance 
    map<pair<Value*, Type*>, Value*> instanceToSubstructInstanceMap; // instanceToInstanceMap; 
    map<Value*, Value*> instanceToParentInstanceMap; 
    map<Value*, Value*> parentInstanceToInstanceMap; 

    Transformer(StructDef structDef, LLVMContext& context): structDef(structDef), context(context) {
        this->originalStructType = structDef.st;       
    }
    
    void transform(vector<int> groupings ){
        errs() << "\n\n-------------------------Transforming Struct Type: " << this->structDef.name<< "---------------------------------";;  
        this->groupings = groupings; 
        this->createMemberIdxMapping(); 
        this->createNewSubstructs(); 

        /*
            a really nasty situation: 
            createMemberIdxMapping and createNewSubstructs will touch: 
                - this->substructMembersTable ()
                - this->parentStructType (this needs to be updated)  
                - this->groupIdxToSubstruct (this needs to have its value updated )
                - this->memberIdxTable (this is fine)
            after create new substructs, we need to then update: 
                - substructs: Some substructs contain a transformed type, need to fix this first 
                - parent: This then needs to be updated AFTER fixing substruct types 
            consider situation when we have 
            struct StructA{
                struct StructB* sb; 
            }
            struct StructB{
                struct StructA* sa;  
            }
            1. transform StructA first
                - StructA_parent
                - SturctA_sub0 
                    - StructB* 
            2. transform StructB 
                - StructB_parent 
                - StructB_sub0 
                    - StructA* 
            3. fix StructA
                - fix StructA_sub0 --> StructA_sub0_fixed  
                    - StructB_parent* 
                - fix StructA_parent --> StructA_parent_fixed 
                    - StructA_sub0_fixed 
            4. fix StructB 
                - fix StructB_sub0 --> StructB_sub0_fixed 
                    - StructA_parent_fixed* 
                - 

            ... let's just ignore circular structs and don't transform them at all. 
            a circular struct is defined as: a struct type that has a type both in its hasMember and isMember 
            e.g. structA's isMember has structB, hasMember also has structB 

            we transform each struct in dependency graph pattern. 
            1. start with structs whose hasMember is empty 
            2. after transforming all of them, replace members of other structs using this type with the parent type,  remove them from hasMember of all other structDefs 
            3. look at which other structs are ready 
            3. transform ready structs 
            4. look at 
        */

        this->createToSubstructMemberMapping(); 
        this->insertNewInstances(); 
        errs() << "\n-----------------------------Original struct member to substruct member map---------------------------\n"; 
        errs() << "Original struct: \n"; 
        this->structDef.st->print(errs()); 
        errs() << "\n"; 
        for(auto& item:originalStructMemberToSubstructMember){
            errs() << " original member idx : "<<item.first<<" \n"; 
            errs() << " substruct type: " << item.second.first;  
            item.second.first->print(errs()); 
            errs() << " substct member idx: "<<item.second.second<<"\n"; 
        }
        errs() << "\n-----------------------------Instance to parent instance map-------------------------\n"; 
        for(auto& item:instanceToParentInstanceMap){
            errs() << "\n Original instance: " << item.first<< " \n"; 
            item.first->print(errs()); 
            errs() << " parent instance: " << item.second << " \n"; 
            item.second->print(errs()); 
            errs() << "\n";  
        }
        this->fixGEPAccesses();         

        // MUST REMOVE LAST 
        this->removeOriginalInstances(); 

        errs() << "\n------------------------------------Member Index table-----------------------------------------"; 
        errs() << "Struct name: " << this->structDef.name << "\n"; 
        for(int i = 0;i<memberIdxTable.size();i++){
            errs() << "original member idx: "<<i<<" group idx: "<<memberIdxTable[i].first<< " substruct member idx: "<<memberIdxTable[i].second<<"\n"; 
        }
    }
    /*
        We can't simply point an original access to member 1 to substruct A member 2 
        this is because there will be recursive structs who has this struct/struct ptr as a member
        We must: 
        1. first get substruct ptr with parent struct access (GEP to parent's substruct field) 
        2. next get substruct member  
        3. replace all uses of original access with 2 
    */
    void removeOriginalInstances(){
        for(auto &item:instanceToParentInstanceMap){
            Instruction* inst = dyn_cast<Instruction>(item.first); 
            inst->eraseFromParent(); 
        }
    }

    void fixGEPAccess(GetElementPtrInst* gepUsage){
        Type* originalGEPStructType = gepUsage->getSourceElementType(); 
        Type* ptrOperandType = gepUsage->getPointerOperandType();
        errs() << "\n---------------------------------------Fixing GEP Usage: "; 
        gepUsage->print(errs()); 
        errs() << "---------------------------------------------------------"; 
        errs() << "\n Struct type: "; 
        originalGEPStructType->print(errs()); 
        errs() << "\nptr operand type: "; 
        ptrOperandType->print(errs()); 
        errs() << "is array ty: " << ptrOperandType->isArrayTy(); 
        errs() << "\n"; 
        unsigned numIndices = gepUsage->getNumIndices(); 
        errs() << "\n Num Indices: " << numIndices << "\n"; 
        
        IRBuilder<> builder(gepUsage->getNextNode()); 
        // check if is array type 
        // if is array type, just replace this gep with parent type 
        if(originalGEPStructType->isArrayTy()){
            errs() << "\n GEP is an array type!!!\n"; 
            ArrayType* arrayType=  dyn_cast<ArrayType>(originalGEPStructType); 
            auto arraySize = arrayType->getArrayNumElements(); 
            errs() << "Array size: "<<arraySize<<"\n"; 
            ArrayType* parentArrayType = ArrayType::get(parentStructType, arraySize); 
            vector<Value*> indices(gepUsage->idx_begin(), gepUsage->idx_end()); 
            for(int i = 0 ;i<indices.size();i++){
                errs() << "indices "<<i<<" : "; 
                indices[i]->print(errs()); 
                errs() << "\n"; 
            }
            
            Value* newParentGEP = builder.CreateGEP(parentArrayType, gepUsage->getPointerOperand(),indices); 
            errs() << "New parent array GEP: "; 
            newParentGEP->print(errs()); 
            errs() << "\n"; 
            gepUsage->replaceAllUsesWith(newParentGEP); 
            // erase original gep usage after replacing it 
            gepUsage->eraseFromParent(); 
        }else{
            errs() << "Number of operands: "<<gepUsage->getNumOperands()<<"\n"; 
            Value* memberIdxVal = gepUsage->getOperand(gepUsage->getNumOperands()-1); 
            ConstantInt* memberIdxInt = dyn_cast<ConstantInt>(memberIdxVal); 
            errs() << "memberIdxInt: "<<memberIdxInt<<"\n"; 
            // if(memberIdxInt){
            if(numIndices==2){ // use # of indices to tell if this is a struct field access 
                errs() << "THIS IS A NORMAL STRUCT FIELD ACCESS!!!\n"; 
                int memberIdx = memberIdxInt->getSExtValue(); 
                errs() << " Member idx: " << memberIdx << "\n";
                if(originalStructMemberToSubstructMember.find(memberIdx)==originalStructMemberToSubstructMember.end()){
                    errs() << "\nMember of index " << memberIdx<<" doesn't exist!!!\n"; 
                    // exit(1); 
                } 
                pair<Type*, int> substructMember = originalStructMemberToSubstructMember[memberIdx]; 
                Type* substructType = substructMember.first; 
                string substructName = substructType->getStructName().str(); 
                int substructMemberIdx = substructMember.second; 
                errs() << "Substruct type: "; 
                substructType->print(errs());
                errs() << "  Member Idx: " << substructMemberIdx << "bro\n";
                Value* ptrOperand = gepUsage->getPointerOperand(); 
                errs() << " Pointer Operand: " << ptrOperand<<" "; 
                ptrOperand->print(errs()); 
                errs() << "\n"; 

                errs() << " substruct type: "; 
                substructType->print(errs()); 

                // create new gep 
                // Value* parentInstance = this->instanceToParentInstanceMap[originalInstance]; 
                Value* parentInstance = ptrOperand; 
                int parentIdx = this->substructToParentIdx[substructType]; 
                errs() << "\nparentInstance: " <<parentInstance<< " parentIdx: "<<parentIdx << " \n"; 
                parentInstance->print(errs()); 
                errs() << "\n"; 
                
                // 1. get parent pointer to substruct 
                Value* gepToParentInstanceSubstructPtrPtr = builder.CreateStructGEP(this->parentStructType, parentInstance, parentIdx, parentStructType->getStructName().str()+"_subptr_ptr"); 
                // this is a ptr to the substruct ptr, 
                errs() << "GEP ptr to parent instance sub field ptr: "; 
                gepToParentInstanceSubstructPtrPtr->print(errs()); 
                errs() << "\n"; 

                // // 2. load the value of the ptr to ptr 
                // Value* gepToParentInstanceSubstructPtr = builder.CreateLoad(PointerType::get(substructType, 0),gepToParentInstanceSubstructPtrPtr, "load_ptr"); 
                // errs() << "Load subfield ptr: "; 
                // gepToParentInstanceSubstructPtr->print(errs()); 
                // errs() << "\n"; 

                // // check if the loaded substruct ptr points to a nullptr 
                // Value* IsNull = builder.CreateICmpEQ(gepToParentInstanceSubstructPtr, ConstantPointerNull::get(PointerType::get(substructType, 0)), "is_null");
                // // branching out here 


                // reload the substruct pointer 
                Value* gepToParentInstanceSubstructPtrReloaded = builder.CreateLoad(PointerType::get(substructType, 0),gepToParentInstanceSubstructPtrPtr,"reload_ptr"); 

                // 3. get substruct field 
                Value* gepToSubstruct = builder.CreateStructGEP(substructType, gepToParentInstanceSubstructPtrReloaded,substructMemberIdx, substructName+"_newptr");  
                errs() << "New GEP to substruct: "; 
                gepToSubstruct->print(errs()); 
                errs() << "\n"; 


                // 4. fix usage on nullptr 
                // Instruction* isNullBlockExit = SplitBlockAndInsertIfThen(IsNull, dyn_cast<Instruction>(gepToParentInstanceSubstructPtrReloaded), false); 
                // IRBuilder<> builder2(&(isNullBlockExit->getParent()->front())); 
                // AllocaInst* substructInstance =  builder2.CreateAlloca(substructType, nullptr); 
                // builder2.CreateStore(substructInstance, gepToParentInstanceSubstructPtrPtr); 
                
                // replace old gep with new gep 
                gepUsage->replaceAllUsesWith(gepToSubstruct); 
                // erase original gep 
                gepUsage->eraseFromParent();
            } else {
                errs() << "THIS IS A DYNAMIC STRUCT ARRAY ACCESS!!!\n"; 
                // for this, we just replace gep with parent struct type 
                vector<Value*> indices(gepUsage->idx_begin(), gepUsage->idx_end()); 
                Value* newParentGEP = builder.CreateGEP(parentStructType, gepUsage->getPointerOperand(), indices); 

                errs() << "New parent GEP: "; 
                newParentGEP->print(errs()); 
                errs() <<"\n"; 

                gepUsage->replaceAllUsesWith(newParentGEP); 
                gepUsage->eraseFromParent(); 
            }
        }  
    }

    void fixGEPAccesses(){
        for(GetElementPtrInst* gepUsage:this->structDef.gepUsages){
            fixGEPAccess(gepUsage); 
        }
    }

    void insertNewInstace(AllocaInst* allocaUsage){
        errs() << "\n---------------------------Inserting New Instance for Alloca usage: "; 
        allocaUsage->print(errs()); 
        errs() << "------------------------------------------------------"; 
        errs() << "\n"; 
        AllocaInst* originalAlloca = allocaUsage; 
        IRBuilder<> builder(originalAlloca->getNextNode()); 
        errs() << "\nIs array allocation: "<<originalAlloca->isArrayAllocation(); 
        errs() << "\n"; 
        unsigned numOperands = allocaUsage->getNumOperands(); 
        errs() << "Num of operands for alloca usage: "<<numOperands<<"\n"; 

        if(allocaUsage->getAllocatedType()->isArrayTy()) {
            errs() << "THIS IS AN FIXED ARRAY ALLOCA USAGE!!!!\n"; 
            Instruction* nextNode = allocaUsage->getNextNode(); 
            // original struct array 
            ArrayType* arrayType = dyn_cast<ArrayType>(allocaUsage->getAllocatedType()); 
            auto arraySize =  arrayType->getArrayNumElements();  
            Type* parentArrayType = ArrayType::get(parentStructType, arraySize); 
            errs() << "new array type: "; 
            parentArrayType->print(errs()); 
            errs() << "\n"; 
            // create allocainst for parent struct array 
            AllocaInst* parentArrayInstance = builder.CreateAlloca(parentArrayType, nullptr, parentStructType->getStructName().str()+"_arr"); 

            llvm::Value* arraySizeValue = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), arraySize);
            errs() << "array size value: \n"; 
            arraySizeValue->print(errs()); 
                // create allocainst for arrays of substructs  
                // we need to point parent's fields to pointers of each substruct 

                errs() << "alloca next node: \n"; 
                nextNode->print(errs()); 
                vector<StructType*> substructTypes; 
                vector<AllocaInst*> substructInstances; 
                vector<int> parentFieldIds; 
                
                for(auto &item:groupIdxToSubstruct) {
                    StructType* substructType = item.second; 
                    Type* substructArrayType = ArrayType::get(substructType, arraySize); 
                    AllocaInst* newSubstructArrayInstance = builder.CreateAlloca(substructArrayType, nullptr, substructType->getStructName().str()+"_arr");
                    int parentFieldIdx = substructToParentIdx[substructType];

                    substructInstances.push_back(newSubstructArrayInstance);     
                    parentFieldIds.push_back(parentFieldIdx); 
                    substructTypes.push_back(substructType); 

                    // just insert ptr for each element for now... 
                        // for(int i = 0;i<arraySize;i++) {
                        //     // This code inserts: parentArrayInstance[i][parentFieldIdx]=&newSubstructArrayInstance[i]; 
                        //     // &newSubstructArrayInstance[i] 
                        //     Value* subStructArrElementPtr = builder.CreateGEP(substructArrayType, newSubstructArrayInstance,{builder.getInt64(0), builder.getInt64(i)}); 
                        //     errs() << "Substruct arr gep: "; 
                        //     subStructArrElementPtr->print(errs()); 
                        //     errs() << "\n"; 
                        //     // &parentArrayInstance[i] 
                        //     Value* parentArrElementPtr = builder.CreateGEP(parentArrayType, parentArrayInstance,{builder.getInt64(0),builder.getInt64(i)}); 
                        //     errs() << "parent arr element ptr: "; 
                        //     parentArrElementPtr->print(errs()); 
                        //     errs() << "\n"; 
                        //     Value* parentElementFieldPtr = builder.CreateStructGEP(parentStructType, parentArrElementPtr, parentFieldIdx); 
                        //     errs() << "parent arr field ptr: "; 
                        //     parentElementFieldPtr->print(errs()); 
                        //     errs() << "\n"; 
                        //     StoreInst* storeSubPtrToParentField = builder.CreateStore(subStructArrElementPtr, parentElementFieldPtr);   
                        //     errs() << "store sub ptr to parent field: "; 
                        //     storeSubPtrToParentField->print(errs()); 
                        //     errs() << "\n"; 
                        // }
                }

                // insert loop to fix things
                for(int i = 0;i<parentFieldIds.size();i++){
                    auto splitPair = SplitBlockAndInsertSimpleForLoop(
                        arraySizeValue, nextNode 
                    ); 
                    Instruction* insertionPoint = splitPair.first; 
                    Value* inductionVariable = splitPair.second; 
                    IRBuilder loopBodyBuilder(insertionPoint->getNextNode()); 
                    AllocaInst* substructInstance = substructInstances[i]; 
                    int parentFieldIdx = parentFieldIds[i]; 
                    StructType* substructType = substructTypes[i]; 
                    ArrayType* substructArrayType = ArrayType::get(substructType, arraySize); 
                    Value* substructArrElementPtr = loopBodyBuilder.CreateGEP(substructArrayType, substructInstance, {loopBodyBuilder.getInt64(0), inductionVariable}); // get ptr to substruct array [i] 
                    Value* parentArrElementPtr = loopBodyBuilder.CreateGEP(parentArrayType, parentArrayInstance,{loopBodyBuilder.getInt64(0), inductionVariable} ); 
                    Value* parentElementFieldPtr = loopBodyBuilder.CreateStructGEP(parentStructType, parentArrElementPtr, parentFieldIdx); 
                    StoreInst* storeSubPtrToParentField = loopBodyBuilder.CreateStore(substructArrElementPtr, parentElementFieldPtr);  
                }

                errs() << "\nTransformed function : \n"; 
                allocaUsage->getParent()->getParent()->print(errs());  

            instanceToParentInstanceMap[originalAlloca] = parentArrayInstance; 
            parentInstanceToInstanceMap[parentArrayInstance] = originalAlloca; 

            // replace original array alloca with parent array instance 
            allocaUsage->replaceAllUsesWith(parentArrayInstance); 
        } else {
            // 1. create instance of parent struct 
            if(!allocaUsage->isArrayAllocation()){
            // if(numOperands==1){ // we can't use numOperands for this, because even if numOperands=2 but the size is 1, this will be accessed like a normal struct 
                errs() << "THIS ALLOCA IS SIMPLE STRUCT ALLOCATION\n"; 
                AllocaInst* parentInstance = builder.CreateAlloca(parentStructType, allocaUsage->getArraySize(), parentStructType->getStructName().str()+"_inst"); 
                instanceToParentInstanceMap[originalAlloca] = parentInstance; 
                parentInstanceToInstanceMap[parentInstance] = originalAlloca; 
                for(auto &item: groupIdxToSubstruct){
                    StructType* substructType = item.second; 
                    string substructName = substructType->getStructName().str(); 
                    // 2. Create instance of substruct 
                    AllocaInst* substructInstance = builder.CreateAlloca(substructType, allocaUsage->getArraySize(), substructName+"_inst");
                    // int substructFieldIdx = splitResult.substructPtrToIdx[item.second]; 
                    // 3. get pointer to the corresponding field on the parent struct  
                    int parentFieldIdx = substructToParentIdx[substructType]; 
                    Value *fieldPtr = builder.CreateStructGEP(parentStructType, parentInstance, parentFieldIdx, substructName+"_fptr"); 
                    // 4. store ptr to substruct instances to the parent struct instance's fields 
                    builder.CreateStore(substructInstance, fieldPtr);
                    
                    // record mapping 
                    this->instanceToSubstructInstanceMap[{originalAlloca, substructType}]=substructInstance; 
                    errs() << "Substruct type: "; 
                    substructType->print(errs()); 
                    errs() << " New Instace: "; 
                    substructInstance->print(errs()); 
                    errs() << "\n";  
                }
                // replace access to original struct with access to parent struct 
                errs() << "Replacing original usage: "; 
                allocaUsage->print(errs()); 
                errs() << " with "; 
                parentInstance->print(errs()); 
                errs() << "\n"; 
                allocaUsage->replaceAllUsesWith(parentInstance); 
                // erase original struct 
                // allocaUsage->eraseFromParent(); 
            }else{
                errs() << "THIS ALLOCA IS STRUCT DYNAMIC ARRAY ALLOCATION\n"; 
                Instruction* nextNode = allocaUsage->getNextNode(); 
                Value* arrSizeVal = allocaUsage->getArraySize(); 
                errs() << "\nArray Size: \n"; 
                arrSizeVal->print(errs()); 
                errs() << "\n"; 

                // simply replace original dynamic array with new dynamic array 
                AllocaInst* parentInstance = builder.CreateAlloca(parentStructType, arrSizeVal, parentStructType->getStructName().str()+"_inst_dyn_arr");
                instanceToParentInstanceMap[originalAlloca] = parentInstance; 
                parentInstanceToInstanceMap[parentInstance] = originalAlloca; 
                
                vector<StructType*> substructTypes; 
                vector<AllocaInst*> substructInstances; 
                vector<int> parentFieldIds; 

                for(auto &item: groupIdxToSubstruct){
                    StructType* substructType = item.second; 
                    string substructName = substructType->getStructName().str(); 
                    // 2. Create instance of substruct 
                    AllocaInst* substructInstance = builder.CreateAlloca(substructType, arrSizeVal, substructName+"_inst");
                    // 3. get pointer to the corresponding field on the parent struct  
                    int parentFieldIdx = substructToParentIdx[substructType]; 
                    substructInstances.push_back(substructInstance); 
                    parentFieldIds.push_back(parentFieldIdx);
                    substructTypes.push_back(substructType);  
                    // record mapping 
                    this->instanceToSubstructInstanceMap[{originalAlloca, substructType}]=substructInstance; 
                }
                
                for(int i = 0;i<parentFieldIds.size();i++){
                    // 4. store ptr to substruct instances to the parent struct instance's fields 
                    // builder.CreateStore(substructInstance, fieldPtr);
                    auto splitPair =  SplitBlockAndInsertSimpleForLoop(
                        arrSizeVal, nextNode 
                    ); 
                    Instruction* insertionPoint = splitPair.first; 
                    errs() << "\nInsertion Point: "; 
                    insertionPoint->print(errs()); 
                    Value* inductionVariable = splitPair.second; 
                    errs() << "\n Induction Variable: "; 
                    inductionVariable->print(errs()); 
                    IRBuilder loopBodyBuilder(insertionPoint->getNextNode()); 
                    AllocaInst* substructInstance = substructInstances[i]; 
                    int parentFieldIdx = parentFieldIds[i]; 
                    StructType* substructType = substructTypes[i]; 
                    auto gepParentElementPtr = loopBodyBuilder.CreateGEP(parentStructType, parentInstance, {inductionVariable}); 
                    auto gepParentElementFPtr = loopBodyBuilder.CreateStructGEP(parentStructType, gepParentElementPtr, parentFieldIdx); 
                    auto gepSubstructElementPtr = loopBodyBuilder.CreateGEP(substructType, substructInstance, {inductionVariable}); 
                    loopBodyBuilder.CreateStore(gepSubstructElementPtr, gepParentElementFPtr);  
                }

                errs() << "\nTransformed function : \n"; 
                allocaUsage->getParent()->getParent()->print(errs());  
                allocaUsage->replaceAllUsesWith(parentInstance); 
                // allocaUsage->eraseFromParent();
            }
        }

    }

    // insert new instance creation on all original struct creation 
    void insertNewInstances(){
        for(AllocaInst* allocaUsage:this->structDef.allocaUsages){
            insertNewInstace(allocaUsage); 
        }
    }

    void createToSubstructMemberMapping(){
        for(int i = 0;i<this->groupings.size();i++){
            int originalMemberIdx = i; 
            pair<int, int> newGroupAndMemberIdx = this->memberIdxTable[originalMemberIdx]; 
            int groupNumber = newGroupAndMemberIdx.first;  
            int substructMemberIdx = newGroupAndMemberIdx.second; 
            StructType* substructType = groupIdxToSubstruct[groupNumber]; 
            this->originalStructMemberToSubstructMember[originalMemberIdx] = {substructType, substructMemberIdx}; 
        }
    }

    
    void createNewSubstructs(){
        string originalStructName = this->originalStructType->getStructName().str(); 
        vector<Type*> substructPtrs; 
        // unordered_map<Type*, int> substructToParentIdx; 
        for(auto &group:this->substructMembersTable){
            int groupIdx = group.first; 
            StructType* subStruct = StructType::create(context, group.second, originalStructName+"_sub"+to_string(groupIdx)); 
            this->substructToParentIdx[subStruct] = substructPtrs.size(); 
            PointerType* substructPtr = PointerType::get(subStruct,0); 
            substructPtrs.push_back(substructPtr); 
            this->groupIdxToSubstruct[groupIdx] = subStruct; 
        }
        StructType* parentStruct = StructType::create(context, substructPtrs, originalStructName+"_parent"); 
        this->parentStructType = parentStruct; 
    }

    void createMemberIdxMapping(){
        vector<pair<int, int>> memberIdxTable(groupings.size()); 
        SubstructMembersTable res; 
        for(int i = 0;i<groupings.size();i++) {
            int group_number = groupings[i]; 
            llvm::Type* member = structDef.members[i];
            int substructMemberIdx = res[group_number].size(); 
            res[group_number].push_back(member);  
            memberIdxTable[i] = {group_number, substructMemberIdx}; 
        }
        this->memberIdxTable = memberIdxTable; 
        this->substructMembersTable = res; 
    }
    


}; 
