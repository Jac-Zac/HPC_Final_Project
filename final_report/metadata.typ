//-------------------------------------
// Document options
//
#let option = (
  type : "final",
  //type : "draft",
  lang : "en",
)
//-------------------------------------
// Optional generate titlepage image
//
#import "@preview/fractusist:0.1.1":*  // only for the generated images

#let titlepage_logo= dragon-curve(
  12,
  step-size: 10,
  stroke-style: stroke(
    //paint: gradient.linear(..color.map.rocket, angle: 135deg),
    paint: gradient.radial(..color.map.rocket),
    thickness: 3pt, join: "round"),
  height: 10cm,
)

//-------------------------------------
// Metadata of the document
//
#let doc= (
  title    : [*Report for High Performance Computing*],
  abbr     : "Prj",
  subtitle : [_Implementation of Parallel Stencil Computation_],
  url      : "https://synd.hevs.io",
  logos: (
    // tp_topleft  : image("resources/img/synd.svg", height: 1.2cm),
    // tp_main : image("resources/img/hei.svg", height: 1.5cm),
    // tp_main : image("resources/img/Cobranding_UniTS_MiGE_Bianco.png", height: 1.5cm),
    tp_main     : titlepage_logo,
    // header      : image("resources/img/project-logo.svg", width: 2.5cm),
  ),
  authors: (
    (
      name        : "Jacopo Zacchigna",
      abbr        : "JZ",
      email       : "jacopo.zacchigna@studenti.units.it",
      // url         : "https://synd.hevs.io",
    ),
    // (
    //   name        : "Axel Amand",
    //   abbr        : "AmA",
    //   email       : "axel.amand@hevs.ch",
    //   url         : "https://synd.hevs.io",
    // ),
  ),
  school: (
    name        : "University Of Trieste",
    major       : "Data Science and Artificial Intelligence ",
  ),
  course: (
    name     : "High Performance Computing",
    url      : "https://github.com/Foundations-of-HPC/High-Performance-Computing-2024",
    prof     : "Stefano Cozzini, Luca Tornatore",
    // class    : [S1f$alpha$],
    semester : "Spring Semester 2024/2025",
  ),
  // keywords : ("Typst", "Template", "Report", "HEI-Vs", "Systems Engineering", "Infotronics"),
  // version  : "v0.1.0",
)

#let date= datetime.today()

//-------------------------------------
// Settings
//
#let tableof = (
  toc: true,
  tof: true,
  tot: true,
  tol: true,
  toe: false,
  maxdepth: 3,
)

#let gloss    = true
#let appendix = false
#let bib = (
  display : true,
  path  : "/tail/bibliography.bib",
  style : "ieee", //"apa", "chicago-author-date", "chicago-notes", "mla"
)
