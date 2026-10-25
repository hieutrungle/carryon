# Living report

Chapters `chapter_0.md`, `chapter_A.md`, `chapter_B.md`, `chapter_C.md`; each milestone updates one.
Figures are cited by ID and come only from `figures/`, which `make figures` rebuilds from `runs/`.
`make report` concatenates the chapters into `report/build/report.md` (a richer build, e.g. pandoc to PDF, is an M0 task).
