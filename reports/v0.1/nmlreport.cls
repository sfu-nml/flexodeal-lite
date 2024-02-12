% NML REPORT TEMPLATE
% LAST UPDATED: JULY 3, 2022 by JAVIER ALMONACID
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nmlreport}[2022/07/01 NML Technical Report Class]

\LoadClass[letterpaper,11pt,oneside,titlepage]{report}
\RequirePackage{mathtools}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{epstopdf}
\RequirePackage{xcolor}
\RequirePackage{lastpage}
\RequirePackage[font=small]{subcaption}

\RequirePackage{caption}
\floatstyle{plaintop}
\restylefloat{table}
%A drawback of this is that you can't have more than one caption per table, i.e. you can't have two different tabulars with two captions side-by-side (question 22751 tex.stackexchange.com)

\RequirePackage{tabularx}
\RequirePackage[left=2cm,top=3cm,right=2cm,bottom=3cm]{geometry} 

%\renewcommand{\normalsize}{\fontsize{10pt}{12pt}\selectfont}
%\setlength{\textwidth}{6.5in}
%\setlength{\textheight}{8in}
\pagenumbering{arabic}

\RequirePackage{sansmathfonts}
\RequirePackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

\RequirePackage{titlesec}
\titleformat{\chapter}[display]
{\normalfont\bfseries\centering}{\chaptertitlename\ \thechapter}{12pt}{\MakeUppercase}

% GET REPORT NUMBER AND TITLE-------------------------

\newcommand{\reportnumber}[1]{\def\@reportnumber{#1}}
\newcommand{\thereportnumber}{\@reportnumber}
\newcommand{\reporttitle}[1]{\def\@reporttitle{#1}}
\newcommand{\thereporttitle}{\@reporttitle}

% GET REPORT AUTHORS, REVISED-BY, APPROVED-BY
\newcommand{\authoredby}[1]{\def\@authoredby{#1}}
\newcommand{\dateauthored}[1]{\def\@dateauthored{#1}}
\newcommand{\revisedby}[1]{\def\@revisedby{#1}}
\newcommand{\daterevised}[1]{\def\@daterevised{#1}}
\newcommand{\approvedby}[1]{\def\@approvedby{#1}}
\newcommand{\dateapproved}[1]{\def\@dateapproved{#1}}


% ----------------------------------------------------

\RequirePackage{fancyhdr}

\fancypagestyle{trheadings}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \fancyhead[L]{NEUROMUSCULAR MECHANICS LABORATORY}
  \fancyhead[R]{Page \thepage/\pageref*{LastPage}}
  \fancyfoot[C]
  {
    \begin{tabularx}{\textwidth}{|>{\centering\arraybackslash}X|c|}\hline 
    \textbf{TECHNICAL REPORT \@reportnumber} & Total Pages\\[0.5em]
    \@reporttitle & \pageref*{LastPage} \\\hline
    \end{tabularx}
  }
}

%\makeatletter
%\renewcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
%                    \thispagestyle{fancy}% original style: plain
%                    \global\@topnum\z@
%                    \@afterindentfalse
%                    \secdef\@chapter\@schapter}
%\makeatother

\def\labelitemi{--}

% ----------------------------------------------

\RequirePackage{hyperref}
\RequirePackage{stmaryrd}
\SetSymbolFont{stmry}{bold}{U}{stmry}{m}{n} % 

% Eliminate ugly boxes around references.
\hypersetup{
    colorlinks,
    linkcolor={red!50!black},
    citecolor={blue!50!black},
    urlcolor={blue!80!black}
}

\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{remark}[theorem]{Remark}

\renewcommand{\theremark}{\thechapter.\arabic{remark}}
\renewcommand{\thetheorem}{\thechapter.\arabic{theorem}}
\renewcommand{\thelemma}{\thechapter.\arabic{lemma}}
\renewcommand{\thecorollary}{\thechapter.\arabic{corollary}}

\renewcommand{\maketitle}{
	\thispagestyle{empty}
	\setlength{\footskip}{36pt}
	{
	\center % Center everything on the page
 
	%-----------------------------
	%	HEADING SECTION
	%-----------------------------
	\vspace*{-6.5em}

	\noindent\begin{minipage}[t]{0.69\textwidth}
	\textbf{SIMON FRASER UNIVERSITY} \\
	\textbf{DEPARTMENT OF BIOMEDICAL PHYSIOLOGY AND KINESIOLOGY} 
	\end{minipage} \hfill
	\begin{minipage}[r]{0.29\textwidth}
	\begin{flushright}
	\vspace*{-0.1cm}
	\includegraphics[width=0.5\textwidth]{logos/sfu-logo.eps}
	\end{flushright}
	\end{minipage}

	%-----------------------------
	%	TITLE SECTION
	%-----------------------------

	\vfill

	\begin{minipage}[c]{0.8\textwidth}
	\centering
	\large \textbf{NEUROMUSCULAR MECHANICS LABORATORY} \\[1em]
	\large \textbf{TECHNICAL REPORT \@reportnumber}\\[2em]
	\includegraphics[width=0.3\textwidth]{logos/nml-logo.png}
	\end{minipage}

	\vspace{8em}

	\begin{minipage}[c]{0.8\textwidth}
	\centering
	\large \textbf{\@reporttitle}
	\end{minipage}

	\vfill
 
	%-----------------------------
	%	AUTHOR SECTION
	%-----------------------------

	\begin{tabularx}{\textwidth}{|X|X|X|}\hline
		\textbf{AUTHOR(S):} & \textbf{REVISED BY}: & \textbf{APPROVED BY:} \\[0.5em]
		\@authoredby & \@revisedby & \@approvedby  \\[0.5em]\hline
		\textbf{LAST UPDATED:} & \textbf{LAST REVISION:} & \textbf{DATE APPROVED:} \\[0.5em]
		\@dateauthored & \@daterevised  & \@dateapproved  \\[0.5em]\hline
	\end{tabularx}\\[2em]

	\begin{tabularx}{\textwidth}{|>{\centering\arraybackslash}X|c|}\hline 
  		\textbf{TECHNICAL REPORT \thereportnumber} & Total Pages\\[0.5em]
  		\thereporttitle & \pageref*{LastPage} \\\hline
	\end{tabularx}
	}

	\vspace{-4.28em}	
}

\AtBeginDocument{
	\pagestyle{trheadings}
	\titleformat{\chapter}
	{\normalfont\Large\bfseries}{\thechapter. }{0pt}{\Large}
	\setlength{\headheight}{14pt}
	\titlespacing*{\chapter}{0pt}{-24pt}{12pt}
	\setlength{\parskip}{1em}
	\titleformat*{\section}{\bfseries}
	\titleformat*{\subsection}{\bfseries}
	\setcounter{page}{1}
}

