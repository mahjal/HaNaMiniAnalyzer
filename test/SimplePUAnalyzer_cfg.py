import FWCore.ParameterSet.Config as cms
# Give the process a name                                                                                                                                                              
process = cms.Process("HaNa")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
#process.MessageLogger.cerr.FwkReport.reportEvery = 1                                                                                                                                  
#process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )                                                                                                      
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.TFileService = cms.Service("TFileService", fileName = cms.string("simple_tree.root") )

# tell the process to run over events (-1) would mean run over                                                                                                                         
#  everything                                                                                                                                                                          
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))

# Tell the process which files to use as the source                                                                                                                                    
process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use                                                                                                                       
                            fileNames = cms.untracked.vstring('file:/eos/home-h/hbakhshi/SWAN_projects/Graph/SingleNeutrino_E-10-gun.root')
)

#file:/eos/home-h/hbakhshi/SWAN_projects/Graph/SingleNeutrino_E-10-gun.root

# Tell the process what filename to use to save the output                                                                                                                             
process.Out = cms.OutputModule("PoolOutputModule",
         fileName = cms.untracked.string ("MyOutputFile.root")
)


process.analyzer1 = cms.EDAnalyzer('SimplePUAnalyzer',
                                 Vertex = cms.PSet( Input = cms.InputTag( "offlineSlimmedPrimaryVertices" ),
                                                    pileupSrc = cms.InputTag("slimmedAddPileupInfo")
                                                ),

                                 Tracks = cms.PSet( Input = cms.InputTag("packedPFCandidates" ) ),
                                 LostTracks = cms.PSet( Input = cms.InputTag("lostTracks" ) ),


                                 sample = cms.string("WJetsMG"),
                                 isData = cms.bool( True ),
                                 SetupDir = cms.string("PUStudies")
                             )


process.p = cms.Path( process.analyzer1 )
