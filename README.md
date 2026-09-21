# Neural Networks: Zero to Hero

This repository is my learning journal while working through Andrej Karpathy's **Neural Networks: Zero to Hero** playlist.

I am implementing the ideas myself, writing notes, and experimenting with the code. The projects are educational exercises, not official implementations of Karpathy's work.

## Projects

| Project                 | Status      | What I am learning                                                  |
| ----------------------- | ----------- | ------------------------------------------------------------------- |
| [micrograd](micrograd/) | In progress | Derivatives, computation graphs, backpropagation, neurons, and MLPs |
| [makemore](makemore/)   | Planned     | Character-level language models and text generation                 |

Each project has its own README with its purpose, setup instructions, progress, and notes.

## Recommended structure

This is one repository with one folder per project:

```text
Neural Network Zero to Hero/
|-- README.md              # Playlist roadmap and repository overview
|-- micrograd/
|   |-- README.md          # Project-specific notes and setup
|   `-- micrograd.ipynb
`-- makemore/
    `-- README.md          # Added as the project develops
```

Keeping everything together makes the learning journey easy to follow, while separate folders keep each project focused. A separate repository would only be useful if a project later becomes an independent, reusable library.

## Setup

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install packages as each project needs them. For the current micrograd notebook:

```bash
pip install numpy matplotlib graphviz jupyter torch
```

Graphviz also needs to be installed on the system. On Ubuntu or Debian:

```bash
sudo apt install graphviz
```

Open notebooks with VS Code or run:

```bash
jupyter notebook
```

## Learning roadmap

- [x] Build scalar automatic differentiation with micrograd.
- [x] Build neurons, layers, and a small MLP.
- [ ] Complete the micrograd experiments and add gradient tests.
- [ ] Build makemore with character-level language models.
- [ ] Study batching, normalization, and training improvements.
- [ ] Implement a small language model and GPT-style components.

The checklist will change as I work through the playlist.

## Attribution

This repository follows the ideas and lessons from Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course and [micrograd](https://github.com/karpathy/micrograd) project.

The code, notes, and experiments here are my own learning work. They are not official Karpathy projects and do not claim ownership of the original material.
