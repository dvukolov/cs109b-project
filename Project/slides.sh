#!/bin/sh

jupyter nbconvert presentation*.ipynb --to slides --TemplateExporter.exclude_input=True --SlidesExporter.reveal_transition=none
# Include slide numbering:
sed -i 's/\(transition: "slide",\)/\1 slideNumber: "c\/t",/' *.slides.html

