executable              = /eos/home-m/mjalalva/CMSSW_12_4_6/src/Haamm/HaNaMiniAnalyzer/test/./job2/SimMiniAOD22/SetupAndRun.sh
output                  = $(ClusterId)_$(ProcId).out
error                   = $(ClusterId)_$(ProcId).err
log                     = $(ClusterId)_$(ProcId).log
+JobFlavour             = "testmatch"
environment             = CONDORJOBID=$(ProcId)
notification            = Error

arguments               = /eos/home-m/mjalalva/CMSSW_12_4_6/src/Haamm/HaNaMiniAnalyzer/test/./job2/.x509up_u154829 slc7_amd64_gcc10 CMSSW_12_4_6 PUGNN SimMiniAOD22 out /eos/home-m/mjalalva/Run1/Nov21/ 1
queue 367

