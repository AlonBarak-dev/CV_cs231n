#!/bin/bash
#NOTE: DO NOT EDIT THIS FILE-- MAY RESULT IN INCOMPLETE SUBMISSIONS
set -euo pipefail

CODE=(
	"CV7062610/layers.py"
	"CV7062610/classifiers/fc_net.py"
	"CV7062610/optim.py"
	"CV7062610/solver.py"
	"CV7062610/classifiers/cnn.py"
)

# these notebooks should ideally
# be in order of questions so
# that the generated pdf is
# in order of questions
NOTEBOOKS=(
	"FullyConnectedNets.ipynb"
	"BatchNormalization.ipynb"
	"Dropout.ipynb"
	"PyTorch.ipynb"
)

FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )

LOCAL_DIR=`pwd`
ASSIGNMENT_NO=2
ZIP_FILENAME="a2.zip"

C_R="\e[31m"
C_G="\e[32m"
C_BLD="\e[1m"
C_E="\e[0m"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") $(find . -name "*.pyx") -x "makepdf.py"

echo -e "### Creating PDFs ###"
python3 makepdf.py --notebooks "${NOTEBOOKS[@]}"

echo -e "### Done! Please submit a2.zip and the pdfs to Gradescope. ###"
