import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as opts
import os
from SamplesPU.Samples import MINIAOD22 as samples  

# Initialize the process
process = cms.Process("HaNa")

# Load configurations
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Output ROOT file
process.TFileService = cms.Service("TFileService", fileName=cms.string("simple_tree.root"))

# Number of events to process
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(20000))

# Source setup
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring())

# Analyzer setup
process.analyzer1 = cms.EDAnalyzer(
    'SimplePUAnalyzer',
    Vertex=cms.PSet(
        Input=cms.InputTag("offlineSlimmedPrimaryVertices"),
        pileupSrc=cms.InputTag("slimmedAddPileupInfo")
    ),
    Tracks=cms.PSet(Input=cms.InputTag("packedPFCandidates")),
    LostTracks=cms.PSet(Input=cms.InputTag("lostTracks")),
    sample=cms.string("WJetsMG"),
    isData=cms.bool(True),
    SetupDir=cms.string("PUStudies")
)

# Command-line options
options = opts.VarParsing('analysis')
options.register('sample', 'SimMiniAOD21', opts.VarParsing.multiplicity.singleton, opts.VarParsing.varType.string, 'Sample to analyze')
options.register('job', 0, opts.VarParsing.multiplicity.singleton, opts.VarParsing.varType.int, 'Job number')
options.register('nFilesPerJob', 1, opts.VarParsing.multiplicity.singleton, opts.VarParsing.varType.int, 'Files per job')
options.register('output', 'out', opts.VarParsing.multiplicity.singleton, opts.VarParsing.varType.string, 'Output file path')
options.parseArguments()

# Select the appropriate sample
theSample = None
for sample in samples:
    print(sample)
    if sample.Name == options.sample:
        theSample = sample
        break

if not theSample:
    raise ValueError(f"Sample {options.sample} not found.")

process.analyzer1.sample = theSample.Name
process.analyzer1.isData = theSample.IsData
print(f'isData = {theSample.IsData}')

# Check if the job number is valid
if not (options.job < theSample.MakeJobs(options.nFilesPerJob, options.output)):
    raise NameError(f"Job {options.job} is not in the list of jobs for sample {options.sample} with {options.nFilesPerJob} files per job.")

# Get the job inputs
job = theSample.Jobs[options.job]

# Handle input files (local and remote)
process.source.fileNames.extend(["file:" + f if not f.startswith("root://") else f for f in job.Inputs])
print("Job Inputs:", job.Inputs)
print("Final Input Files for cmsRun:", process.source.fileNames)

# Set output file
process.TFileService.fileName = job.Output

# Set up the path
process.p = cms.Path(process.analyzer1)
