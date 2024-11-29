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
#include "llvm/Support/CommandLine.h"
// #include "llvm/Analysis/BlockFrequencyAnalysis.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream> 
#include <fstream> 
#include "json.hpp" 
#include "Analyzer.hpp" 
/* *******Implementation Ends Here******* */
#define HOTTEST_LOOP_CNT 10 // only the 10 hottest loops are considered 
using namespace llvm;
using namespace std;   
static llvm::cl::opt<std::string> StructOption(
  "struct-opt", 
  llvm::cl::desc("Test"), 
  llvm::cl::value_desc("test"), 
  llvm::cl::init("wtf") 
); 

static llvm::cl::opt<int> StructOption_LoopCnt( 
  "loop-cnt", 
  llvm::cl::desc("L hottest loops"), 
  llvm::cl::value_desc("test"), 
  llvm::cl::init(10)  
); 


static llvm::cl::opt<std::string> StructOption_FeatureMode( 
  "feature-mode", 
  llvm::cl::desc("Feature Mode"), 
  llvm::cl::value_desc("test"), 
  llvm::cl::init("LOOP")   
); 
static int getFeatureMode(){
  if(StructOption_FeatureMode=="LOOP") return MODE_LOOP; 
  return MODE_BB; 
}; 
namespace {
  struct StructAnalysisPass : public PassInfoMixin<StructAnalysisPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM){ 
      errs() << "-----------------------------------------------------ANALYSIS PASS------------------------------------------------------------\n";
      errs() << StructOption << "\n"; 
      Analyzer analyzer(M, MAM, StructOption_LoopCnt, getFeatureMode());   

      return PreservedAnalyses::none(); 
    }
  }; 


struct StructSplittingPass : public PassInfoMixin<StructSplittingPass> {

    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
      errs() << "-----------------------------------------------------TRANSFORM PASS------------------------------------------------------------\n";
      Analyzer analyzer(M, MAM, StructOption_LoopCnt, getFeatureMode());   
      StructGroupingTable groupingTable =  analyzer.loadStructGroupingTable(GROUPING_FNAME); 
      analyzer.printStructGroupingTable(groupingTable); 
      analyzer.transform(); 

      return PreservedAnalyses::none(); 
    }
  }; 
};

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "StructPass", "v0.1",
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM, // FunctionPassManager &FPM,
        ArrayRef<PassBuilder::PipelineElement>) {
          if(Name=="struct-analysis"){
            // FPM.addPass(StructAnalysisPass()); 
            MPM.addPass(StructAnalysisPass()); 
            return true; 
          }
          if(Name=="struct-splitting"){
            // FPM.addPass(StructSplittingPass()); 
            MPM.addPass(StructSplittingPass()); 
            return true; 
          }
          return false;
        }
      );
    }
  };
}