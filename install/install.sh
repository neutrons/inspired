#!/bin/bash

sdir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "#!/bin/bash" > inspired
echo "python ${sdir::-7}gui/inspired.py" >> inspired
chmod 755 inspired
edir=$( which python )
mv inspired ${edir::-6}
