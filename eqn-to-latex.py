def convert_to_latex(input_file, output_file):
    with open(input_file, 'r') as file:
        expression = file.read()

    # Break the expression into lines (you can customize this logic)
    # Here, we just add a line break every 100 characters for demonstration
    max_line_length = 80
    lines = [expression[i:i + max_line_length] for i in range(0, len(expression), max_line_length)]

    # Generate LaTeX code
    latex_code = '\\begin{align}\n'
    latex_code += '\n'.join(f'  {line} \\\\' for line in lines)
    latex_code += '\n\\end{align}'

    # Write to output file
    with open(output_file, 'w') as file:
        file.write(latex_code)

# Usage
convert_to_latex('acc-eqn.txt', 'acc-eqn-latex.tex')
