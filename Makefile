NAME = matlab-lbfgs

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex
	pdflatex $(NAME)
	pdflatex $(NAME)

clean:
	rm -rf *.log *.aux *.out $(NAME).pdf
