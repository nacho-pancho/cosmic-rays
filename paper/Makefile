#
## Options
# 
TEXOPT = -interaction nonstopmode -file-line-error-style
TEXIN := .:./inc:
TEXFO := ./inc/tfm/::
#
# various filenames
#
TEX = paper.tex #$(wildcard *.tex)
BIB = $(patsubst %.tex,%.bib,$(TEX))
PDF = $(patsubst %.tex,%.pdf,$(TEX))
AUX = $(patsubst %.tex,%.aux,$(TEX))
BBL = $(patsubst %.tex,%.bbl,$(TEX))
BASE = $(patsubst %.tex,%   ,$(TEX))
#
# we want the pdf
#
all: $(PDF) 

$(AUX): $(TEX)
	echo "----- FIRST PASS -----"
	TEXFONTS=$(TEXFO) TEXINPUTS=$(TEXIN) pdflatex $(TEXOPT) -draftmode $<

$(PDF): $(AUX) $(BBL) Makefile
	echo "----- SECOND PASS ----"
	TEXFONTS=$(TEXFO) TEXINPUTS=$(TEXIN) pdflatex $(TEXOPT) -draftmode -interaction batchmode $(BASE)
	echo "----- FINAL PASS -----" 
	TEXFONTS=$(TEXFO) TEXINPUTS=$(TEXIN) pdflatex $(TEXOPT) -interaction batchmode $(BASE)

#
# the bibliography depends on aux and bib files
#
$(BBL): $(BIB) $(AUX)
	echo "----- BIBLIOGRAPHY -----"
	-bibtex $(AUX)

#
# cleanup targets
#
clean:
	rm -f *.aux *.log *.out *.ps *.toc *.nav *.snm *.dvi *.bbl *.blg *.idx *.glo *.ist *~ *bak


purge: clean
	-rm -f *.pdf

