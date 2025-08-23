#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#import "/tail/bibliography.typ": *
#import "/tail/glossary.typ": *
#show:make-glossary
#register-glossary(entry-list)

//-------------------------------------
// Template config
//
#show: report.with(
  option: option,
  doc: doc,
  date: date,
  tableof: tableof,
)

// Increase font size
#set text(size: 12pt)
// Set links to be underlineeed
#show link: underline
// Change line spacing
#set par(leading: 1em)
// Keep other spacing more sensible
#show par: set par(spacing: 1.5em)

//-------------------------------------
// Content
//
#include "/main/01-intro.typ"
#include "/main/02-vm_setup.typ"
#include "/main/06-conclusion.typ"
// #include "/main/to_know.typ"

#heading(numbering:none, outlined: false)[] <sec:end>

//-------------------------------------
// Glossary
// //
// #make_glossary(gloss:gloss, title:i18n("gloss-title"))

//-------------------------------------
// Bibliography
//
#make_bibliography(bib:bib, title:i18n("bib-title"))

//-------------------------------------
// Appendix
//
#if appendix == true {[
  #pagebreak()
  #counter(heading).update(0)
  #set heading(numbering:"A")
  = #i18n("appendix-title") <sec:appendix>
  //#include "tail/a-appendix.typ"
]}
