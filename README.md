<snippet>
  <content><![CDATA[
# ${1:QCode}

First version of QCode, an educational software for quantum chemistry calculations based on GSL (GNU Scientific Library). The code is not optimized, but it should work.
The reason I am coding this is for personal practice, but it would be great if someone could also benefit from it.

## Installation
Run the makefile.
NOTE: Remember to add the absolute path of GSL to the Makefile, before trying to compile.

## Usage

Documentation is not available yet, but you can see the example files to get an idea on how to run sample calculations. The example is for H2O with STO-3G.

Currently supporting: HF + DIIS\s\s 
                      MP2
                      CCSD + DIIS
                      CCSD(T)
                      CIS
                      RPA
but by providing the one and two electron integrals.

## Credits

The idea is taken from prof. Crawford.


]]></content>
</snippet>
