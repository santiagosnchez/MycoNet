
jobscript="sim_${sqlen}_${samples}_${species}_${div}_${poly}.sh"

python training_on_sim.py $sqlen $samples $species $div $poly 50

wd=`pwd`
ssh mist-login01 " cd $wd && sbatch $jobscript "
