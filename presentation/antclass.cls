% This material is declared a work of the United States Government and is not
% subject to copyright protection in the United States.
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and sell copies
% of the Software, and to permit persons to whom the Software is furnished to do
% so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
% ------------------------------------------------------------------------------
% Author: David Woodburn
% Description: class lecture presentation template
% ------------------------------------------------------------------------------
% The dimensions of the text box are
%
%   Columns |  Width  | Height
%   ------- | ------- | ------
%      1    | 13.5 cm |  7 cm
%      2    | 6.25 cm |  7 cm

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{antclass}[2025/03/31 ANT Center Class]
\def\@version{20250331}

% Process options.
\newif\ifstandard\standardfalse
\newif\ifdefineul\defineultrue
\DeclareOption{standard}{\standardtrue}
\DeclareOption{wide}{\standardfalse}
\DeclareOption{noul}{\defineulfalse}
\DeclareOption*{\PackageWarning{antbrief}{Unknown '\CurrentOption'}}
\ProcessOptions\relax
\ifstandard
    \LoadClass[10pt]{article}
\else
    \LoadClass[12pt]{article}
\fi

% Packages
\RequirePackage[T1]{fontenc} % use 8-bit fonts
\RequirePackage{geometry} % page dimensions and margins
\RequirePackage{fancyhdr} % custom headers
\RequirePackage{float} % custom floats
\RequirePackage{flafter} % (builtin) floats after their definitions
\RequirePackage[table]{xcolor} % custom table cell colors
\RequirePackage{pgfplots} % includes tikz, xcolor, graphicx
\RequirePackage{adjustbox} % automatically scale to fit
\RequirePackage{sourcesanspro} % custom default font
\RequirePackage[scaled=0.89]{beramono} % custom teletype font
\RequirePackage{amsmath, amsfonts, amssymb} % improved math
\RequirePackage{bm} % bold math (must be after amsmath)
\RequirePackage{cancel} % strike-through in equations
\RequirePackage{listings} % code and algorithms
\RequirePackage{matlab-prettifier} % better MATLAB code formatting
\RequirePackage{upquote} % upright single quotes in verbatim
\RequirePackage{array} % custom column alignment types
\RequirePackage[absolute]{textpos} % free placement of objects
\RequirePackage{enumitem} % custom lists and list spacing
\RequirePackage{hyperref} % hyperlinks

% TikZ libraries
\usetikzlibrary{math} % general math calculations in TikZ

% PGF Plot configuration
\pgfplotsset{compat=newest}

% ----------------------
% Added content features
% ----------------------

% Standard math macros
\ifdefineul
    \newcommand{\ul}[1] % vectors
        {{}\mkern1mu\underline{\mkern-1mu#1\mkern-1mu}\mkern1mu}
\fi
\DeclareSymbolFont{euler}{U}{eur}{m}{n}     % symbol font
\DeclareMathSymbol{\PI}{\mathord}{euler}{25} % up pi

% Convenience macros
\DeclareMathOperator{\sgn}{sgn}             % sign function
\DeclareMathOperator{\tr}{tr}               % trace of a matrix
\DeclareMathOperator{\adj}{adj}             % adjugate of a matrix
\DeclareMathOperator*{\argmax}{arg\,max}    % argmax operator
\DeclareMathOperator*{\argmin}{arg\,min}    % argmin operator
\newcommand{\abs}[1]{\left\lvert#1\right\rvert} % absolute value
\newcommand{\norm}[1]{\left\lVert#1\right\rVert} % vector or matrix norm
\newcommand{\ssm}[1]{[#1]_{\rule{0pt}{1.38ex}\times}} % skew symmetric matrix
\newcommand{\diff}[2]{\frac{\mathrm{d}{#1}}{\mathrm{d}{#2}}} % derivative
\newcommand{\piff}[2]{\frac{\partial{#1}}{\partial{#2}}} % partial derivative
\newcommand{\EE}[2]{\ensuremath{{{#1}\!\times\!10^{#2}}}} % scientific notation
\newcommand{\rom}[1]{\uppercase\expandafter{\romannumeral #1\relax}}

% Upright math letters
\newcommand{\upc}[1][]{\mathrm{c}_{#1}}     % abbreviated cosine
\newcommand{\ups}[1][]{\mathrm{s}_{#1}}     % abbreviated sine
\newcommand{\upt}[1][]{\mathrm{t}_{#1}}     % abbreviated tangent
\newcommand{\upd}{\mathrm{d}}               % differential operator
\newcommand{\upe}{\mathrm{e}}               % Euler's number
\newcommand{\upi}{\mathrm{i}}               % imaginary number
\newcommand{\upj}{\mathrm{j}}               % imaginary number
\newcommand{\upk}{\mathrm{k}}               % quaternion number

% Increase maximum number of matrix columns.
\setcounter{MaxMatrixCols}{20}

% Define custom table column types (array package) and spacings.
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}
\renewcommand{\arraystretch}{1.3} % spacing between matrix rows
\setlength{\tabcolsep}{1ex} % spacing between columns of a table

% Custom colors (xcolor package)
\definecolor{blue}{RGB}{0,0,255}        % 0000ff
\definecolor{azure}{RGB}{0,128,255}     % 0080ff
\definecolor{cyan}{RGB}{0,209,209}      % 00d1d1
\definecolor{green}{RGB}{15,180,0}      % 0fb400
\definecolor{lime}{RGB}{180,216,0}      % b4d800
\definecolor{yellow}{RGB}{255,191,0}    % ffbf00
\definecolor{orange}{RGB}{255,128,0}    % ff8000
\definecolor{red}{RGB}{255,0,0}         % ff0000
\definecolor{magenta}{RGB}{255,0,255}   % ff00ff
\definecolor{purple}{RGB}{128,0,255}    % 8000ff
\definecolor{colorA}{RGB}{0,38,76}
\definecolor{colorB}{RGB}{4,47,89}
\definecolor{colorC}{RGB}{10,56,102}
\definecolor{colorD}{RGB}{17,66,115}
\definecolor{colorE}{RGB}{25,76,127}
\definecolor{colorH}{RGB}{58,112,166}

% Clickable mailto email addresses
\newcommand{\mailto}[1]{\href{mailto:#1}{\nolinkurl{#1}}}

% Horizontal line with middle text
\newsavebox{\@btext}
\newlength{\@bwidth}
\newcommand{\breaktext}[2][colorH]{%
    \sbox{\@btext}{#2}%
    \setlength{\@bwidth}{0.5\textwidth}%
    \addtolength{\@bwidth}{-0.5\wd\@btext}%
    \addtolength{\@bwidth}{-1ex}%
    \par{\color{#1!20}\rule[0.5ex]{\@bwidth}{1pt}}%
    \hfill{\usebox{\@btext}}\hfill%
    {\color{#1!20}\rule[0.5ex]{\@bwidth}{1pt}}\par}

% Vertical and horizontal centering
\newenvironment{Center}
    {\topskip0pt\par\vspace*{\fill}\vspace*{-\parskip}\vspace*{-\topsep}%
    \begin{minipage}{\textwidth}\centering}
    {\end{minipage}\vspace*{\fill}\vspace*{-\topsep}\par}

% Arbitrary object placement (textpos package)
\newcommand{\pos}[4][]{%
    \if\relax\detokenize{#1}\relax
        \sbox{\@tempboxa}{#4}
        \begin{textblock*}{\wd\@tempboxa}(#2,#3)
            \usebox{\@tempboxa}
        \end{textblock*}
    \else
        \begin{textblock*}{#1}(#2,#3)
            #4
        \end{textblock*}
    \fi}

% Colored text box: \shadedbox[color]{text}
\definecolor{shade}{RGB}{213,227,255}
\newcommand{\shadedbox}[2][shade]{\setlength{\fboxsep}{5pt}%
    \hspace*{-5pt}%
    \colorbox{#1}{\parbox{\columnwidth}{#2}}%
    \hspace*{-5pt}}

% -------------------------------------
% Custom code styles (listings package)
% -------------------------------------

% Pseudocode
\lstdefinestyle{pseudostyle}{
    xleftmargin=30pt, xrightmargin=0pt,
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=10pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    basicstyle=\linespread{1.1}\ttfamily\small,
    commentstyle=\color{gray}\itshape,
    stringstyle=\color{orange},
    columns=flexible}
\lstdefinelanguage{pseudolang}{
    keywords={function, for, while, do, break, continue,
        if, then, else, end, in, to, until, return},
    keywordstyle=\color{black}\bfseries,
    morecomment=[l]{\%\ },
    morestring=[b]",
    morestring=[b]',
    mathescape=true,
    escapechar=`}
\lstnewenvironment{pseudocode}[1][1]{%
    \lstset{language=pseudolang, style=pseudostyle, firstnumber=#1}}{}
\lstnewenvironment{pseudocodetight}[1][1]{%
    \lstset{language=pseudolang, style=pseudostyle, firstnumber=#1,
        xleftmargin=15pt, numbersep=10pt, rulecolor=\color{white}}}{}

% Python code
\lstdefinestyle{pythonstyle}{
    xleftmargin=40pt, xrightmargin=0pt,
    frame=leftline, framesep=9pt, framerule=2pt, rulecolor=\color{azure!20},
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=20pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    basicstyle=\ttfamily\footnotesize,
    commentstyle=\color{gray},
    stringstyle=\color{orange},
    keywordstyle=[1]\color{azure},   % keywords
    keywordstyle=[2]\color{azure},   % builtin functions
    keywordstyle=[3]\color{purple},  % NumPy
    keywordstyle=[4]\color{magenta}} % MatplotLib
\lstdefinelanguage{pythonlang}{
    keywords={and, as, assert, break, class, continue, def, del, elif, else,
        except, False, finally, for, from, global, if, import, in, is, lambda,
        None, nonlocal, not, or, pass, raise, return, True, try, while, with,
        yield},
    morekeywords=[2]{abs, aiter, all, any, anext, ascii, bin, bool, breakpoint,
        bytearray, bytes, callable, chr, classmethod, compile, complex, delattr,
        dict, dir, divmod, enumerate, eval, exec, exit, filter, float, format,
        frozenset, getattr, globals, hasattr, hash, help, hex, id, input, int,
        isinstance, issubclass, iter, len, list, locals, map, max, memoryview,
        min, next, object, oct, open, ord, quit, pow, print, property, range,
        repr, reversed, round, set, setattr, slice, sorted, staticmethod, str,
        sum, super, tuple, type, vars, zip},
    morekeywords=[3]{DataSource, LinAlgError, MachAr, NDArrayOperatorsMixin,
        NumpyVersion, abs, absolute, add, all, allclose, amax, amin, angle,
        any, append, apply_along_axis, apply_over_axis, arange, arccos, arccosh,
        arcsin, arcsinh, arctan, arctan2, arctanh, argmax, argmin, argpartition,
        argsort, argwhere, around, array, array2string, array_equal,
        array_equiv, array_repr, array_split, array_str, arraysetops,
        asanyarray, asarray, asarray_chkfinite, ascontiguousarray, asfarray,
        asfortranarray, asmatrix, asscalar, atleast_1d, atleast_2d, atleast_3d,
        average, bartlett, base_repr, binary_repr, bincount, bitwise_and,
        bitwise_or, bitwise_xor, blackman, block, bmat, broadcast,
        broadcast_arrays, broadcast_shapes, broadcast_to, busday_count,
        busday_offset, busdaycalendar, byte_bounds, c_, can_cast, capitalize,
        cbrt, ceil, center, chararray, chebyshev, cholesky, choose, cla, clip,
        column_stack, common_type, compare_chararrays, compress, concatenate,
        cond, conj, conjugate, convolve, copy, copysign, copyto, corrcoef,
        correlate, cos, cosh, count, count_nonzero, cov, cross, cumprod, cumsum,
        datetime_as_string, datetime_data, decode, deg2rad, degrees, delete,
        deprecate, deprecate_with_doc, det, diag, diag_indices,
        diag_indices_from, diagflat, diagonal, diff, digitize, disp, divide,
        divmod, dot, dsplit, dstack, dtype, ediff1d, eig, eigh, eigs, eigvals,
        eigvalsh, einsum, einsum_path, empty, empty_like, encode, endswith,
        equal, errstate, exp, exp2, expand_dims, expandtabs, expm1, extract,
        eye, fabs, fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn,
        ifftshift, ihfft, irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftn,
        fill_diagonal, find, find_common_type, finfo, fix, flat, flatiter,
        flatnonzero, flatten, flip, fliplr, flipud, float_power, floor,
        floor_divide, fmax, fmin, fmod, format_float_positional,
        format_float_scientific, format_parser, frexp, frombuffer, fromfile,
        fromfunction, fromiter, frompyfunc, fromregex, fromstring, fsolve, full,
        full_like, gcd, genfromtxt, geomspace, get_include, get_printoptions,
        getbufsize, geterr, geterrcall, geterrobj, gradient, greater,
        greater_equal, hamming, hanning, heaviside, hermite, hermite_e,
        histogram, histogram2d, histogram_bin_edges, histogramdd, hsplit,
        hstack, hypot, i0, identity, identiy, iinfo, imag, in1d, index, indices,
        inner, insert, interp, intersect1d, inv, invert, is_busday, isalnum,
        isalpha, isclose, iscomplex, iscomplexobj, isdecimal, isdigit, isfinite,
        isfortran, isin, isinf, islower, isnan, isnat, isneginf, isnumeric,
        isposinf, isreal, isrealobj, isscalar, issctype, isspace, issubclass_,
        issubdtype, issubsctype, istitle, isupper, ix_, join, kaiser, kron,
        laguerre, lcm, ldexp, left_shift, legendre, less, less_equal, lexsort,
        linspace, ljust, load, loadtxt, log, log10, log1p, log2, logaddexp,
        logaddexp2, logical_and, logical_not, logical_or, logical_xor, logspace,
        lower, lstrip, lstsq, mask_indices, mat, matmul, matrix_power,
        matrix_rank, max, maximum, maximum_sctype, may_share_memory, mean,
        median, memmap, meshgrid, mgrid, min, min_scalar_type, minimum,
        mintypecode, mod, modf, moveaxis, msort, multi_dot, multiply,
        nan_to_num, nanargmax, nanargmin, nancumprod, nancumsum, nanmax,
        nanmean, nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd,
        nansum, nanvar, ndenumerate, ndindex, nditer, negative, nested_iters,
        nextafter, nonzero, norm, not_equal, obj2sctype, ode, ogrid, ones,
        ones_like, outer, packbits, pad, partition, pcolor, percentile, pi,
        piecewise, pinv, place, polyfit, polynomial, polyval, positive, power,
        printoptions, prod, promote_types, ptp, put, put_along_axis, putmask,
        qr, quad, quantile, r_, rad2deg, radians, rand, randn, ravel,
        ravel_multi_index, real, real_if_close, reciprocal, remainder, repeat,
        replace, repmat, require, reshape, resize, result_type, rfind,
        right_shift, rindex, rint, rjust, roll, rollaxis, rot90, round_, round,
        row_stack, rpartition, rsplit, rstrip, s_, save, savetxt, savez,
        savez_compressed, sctype2char, searchsorted, select, set_printoptions,
        set_string_function, setbufsize, setdiff1d, seterr, seterrcall,
        seterrobj, setxor1d, shape, shares_memory, show_config, sign, signbit,
        sin, sinc, sinh, slogdet, solve, sort, sort_complex, spacing, split,
        splitlines, sqrt, square, squeeze squeeze, stack, startswith, std,
        str_len, strip, subtract, sum, svd, swapaxes, swapcase, take,
        take_along_axis, tan, tanh, tensordot, tensorinv, tensorsolve, tile,
        title, tofile, tolist, trace, translate, transpose, trapz, tri, tril,
        tril_indices, tril_indices_from, trim_zeros, triu, triu_indices,
        triu_indices_from, true_divide, trunc, typename, union1d, unique,
        unpackbits, unravel_index, unwrap, upper, vander, var, vdot, vectorize,
        vsplit, vstack, where, who, xlabel, ylabel, zeros, zeros_like, zfill},
    morekeywords=[4]{acorr, angle_spectrum, annotate, arrow, autoscale, autumn,
        axes, axhline, axhspan, axis, axline, axvline, axvspan, bar, bar_label,
        barbs, barh, bone, box, boxplot, broken_barh, cla, clabel, clf, clim,
        close, cohere, colorbar, connect, contour, contourf, cool, copper, csd,
        delaxes, disconnect, draw, draw_if_interactive, errorbar, eventplot,
        figimage, figlegend, fignum_exists, figtext, figure, fill, fill_between,
        fill_betweenx, findobj, flag, gca, gcf, gci, get,
        get_current_fig_manager, get_figlabels, get_fignums, get_plot_commands,
        getp, ginput, gray, grid, hexbin, hist, hist2d, hlines, hot, hsv,
        imread, imsave, imshow, inferno, install_repl_displayhook, ioff, ion,
        isinteractive, get, legend, locator_params, loglog, magma,
        magnitude_spectrum, margins, matshow, minorticks_off, minorticks_on,
        new_figure_manager, nipy_spectral, pause, pcolor, pcolormesh,
        phase_spectrum, pie, pink, plasma, plot, plot_date, polar, prism, psd,
        quiver, quiverkey, rc, rc_context, rcdefaults, rgrids, savefig, sca,
        scatter, sci, semilogx, semilogy, set_cmap, setp, show, specgram,
        spring, spy, stackplot, stairs, stem, step, streamplot, subplot,
        subplot2grid, subplot_mosaic, subplot_tool, subplots, subplots_adjust,
        summer, suptitle, switch_backend, table, text, thetagrids, tick_params,
        ticklabel_format, tight_layout, title, tricontour, trcontourf,
        tripcolor, triplot, twinx, twiny, uninstall_repl_displayhook,
        violinplot, viridis, vlines, waitforbuttonpress, winter, xcorr, xkcd,
        xlabel, xlim, xscale, xticks, ylabel, ylim, yscale, yticks},
    morecomment=[l]{\#\ },
    morecomment=[s]{"""}{"""},
    morestring=[b]",
    morestring=[b]',
    prebreak={ \char92},
    escapechar=`}
\lstnewenvironment{python}[1][1]{%
    \lstset{language=pythonlang, style=pythonstyle, firstnumber=#1}}{}
\newcommand*{\inputpython}[3]{\lstinputlisting[linerange={{#2}-{#3}},
    firstnumber=#2, language=pythonlang, style=pythonstyle]{#1}}
\newcommand*{\py}{\lstinline[language=pythonlang, basicstyle=\ttfamily]}

% MATLAB code
\lstdefinestyle{matlabstyle}{
    style=Matlab-editor,
    basicstyle=\ttfamily\footnotesize,
    xleftmargin=40pt, xrightmargin=0pt,
    frame=leftline, framesep=9pt, framerule=2pt, rulecolor=\color{green!20},
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=20pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    prebreak={ ...},
    escapechar=`}
\lstnewenvironment{matlab}[1][1]{%
    \lstset{style=matlabstyle, firstnumber=#1}}{}
\newcommand*{\inputmatlab}[3]{\lstinputlisting[
    style=matlabstyle, linerange={{#2}-{#3}}, firstnumber=#2]{#1}}
\newcommand*{\mat}{\lstinline[style=Matlab-editor, basicstyle=\ttfamily]}

% R code
\lstdefinestyle{rstyle}{
    xleftmargin=40pt, xrightmargin=0pt,
    frame=leftline, framesep=9pt, framerule=2pt, rulecolor=\color{purple!20},
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=20pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    basicstyle=\linespread{1.2}\ttfamily\footnotesize,
    commentstyle=\color{green},
    stringstyle=\color{red},
    keywordstyle=\color{azure}}
\lstdefinelanguage{rlang}{
    language=R,
    morekeywords={TRUE, FALSE},
    deletekeywords={data, frame, length, as, character, gamma, beta},
    prebreak={},
    escapechar=`}
\lstnewenvironment{rlang}[1][1]{%
    \lstset{language=rlang, style=rstyle, firstnumber=#1}%
    \colorlet{ruleColor}{purple}}{}
\newcommand*{\inputrlang}[3]{\lstinputlisting[linerange={{#2}-{#3}},
    firstnumber=#2, language=rlang, style=rstyle]{#1}}
\newcommand*{\rlng}{\lstinline[language=rlang, basicstyle=\ttfamily]}

% Line highlighting
\newcommand{\HL}{\tikz[overlay]{\draw[black, line width=2pt]
    (-10pt,-4pt) -- (-10pt,10pt);}}

% ---------------------------
% General document formatting
% ---------------------------

% Default font family
% Font set by sourcesanspro package
\renewcommand\familydefault{\sfdefault}

% Default plot settings
\pgfplotsset{%
    tick label style={font=\scriptsize\sffamily},
    every axis label={font=\sffamily},
    legend style={font=\tiny\sffamily},
    label style={font=\sffamily},
    nodes near coords style=
        {font=\scriptsize,/pgf/number format/assume math mode}
}

% Define margins.
\ifstandard
    \newlength{\@xw}\setlength{\@xw}{6cm}   % half page width
\else
    \newlength{\@xw}\setlength{\@xw}{8cm}   % half page width
\fi
\newlength{\@yh}\setlength{\@yh}{4.5cm}     % half page height
\newlength{\@mgnT}\setlength{\@mgnT}{1.5cm} % top margin
\newlength{\@mgnB}\setlength{\@mgnB}{0.499cm} % bottom margin
\newlength{\@mgnL}\setlength{\@mgnL}{1.5cm} % left margin
\newlength{\@mgnC}\setlength{\@mgnC}{1cm}   % column separation
\newlength{\@mgnR}\setlength{\@mgnR}{1cm}   % right margin
\newlength{\@mgnS}\setlength{\@mgnS}{\@mgnT} % head separation
\addtolength{\@mgnS}{-1cm}\addtolength{\@mgnS}{0.348cm}

% Page geometry
\geometry{
    paperwidth=2\@xw, paperheight=2\@yh,
    top=\@mgnT, bottom=\@mgnB,
    left=\@mgnL, right=\@mgnR,
    columnsep=\@mgnC,
    headheight=1cm, headsep=\@mgnS}
\setlength{\columnseprule}{0.1pt}

% Derived dimensions
\newlength{\@xShift}\setlength{\@xShift}{\@xw} % full-page x shift
\addtolength{\@xShift}{-\@mgnL}
\newlength{\@yShift}\setlength{\@yShift}{\@mgnT} % full-page y shift
\addtolength{\@yShift}{-4.078cm}
\newlength{\@xHead}\setlength{\@xHead}{-0.5\@mgnL} % header x position
\addtolength{\@xHead}{0.5\@mgnR}
\newlength{\@yHead}\setlength{\@yHead}{-4cm} % header y position

% Paragraph parameters
\setlength{\parindent}{0pt} % indentation of first line of paragraphs
\setlength{\parskip}{1.2ex plus 6pt} % space between paragraphs
\setlength{\emergencystretch}{\hsize} % overfull avoidance
\overfullrule=1mm % overfull black bands
\raggedbottom % page-ending empty space allowed
\DeclareRobustCommand\justifying{%
    \let\\\@normalcr
    \@rightskip\z@skip \rightskip\@rightskip
    \leftskip\z@skip
    \finalhyphendemerits=5000
    \parfillskip=0pt plus 1fil}

% Reduce list spacing (enumitem package).
\setlist{
    itemindent=0pt,
    labelsep=0.2cm,
    labelwidth=0.4cm,
    leftmargin=0.6cm,
    topsep=0pt,
    parsep=0.5\parskip,
    itemsep=0.5\parskip,
    before=\raggedright}

% Reduce spacing of other environments (enumitem package).
\setlist[trivlist]{topsep=0pt}

% Hyperlink styling
\hypersetup{
    colorlinks=true,
    linkcolor=colorE,
    citecolor=colorE,
    urlcolor=colorE,
    hypertexnames=false}
\def\UrlFont{\sffamily\itshape}

% Remove pdf image warnings.
\pdfsuppresswarningpagegroup=1

% Use French spacing.
\frenchspacing

% Pretitle
\newif\ifpretitle\pretitlefalse
\def\@pretitle{}
\newcommand{\pretitle}[1]{\pretitletrue\def\@pretitle{#1}}
% Title
\newif\iftitle\titlefalse
\def\@title{}
\renewcommand{\title}[1]{\titletrue\def\@title{#1}}
% Subtitle
\newif\ifsubtitle\subtitlefalse
\def\@subtitle{}
\newcommand{\subtitle}[1]{\subtitletrue\def\@subtitle{#1}}
% Author
\newif\ifauthor\authorfalse
\def\@author{}
\renewcommand{\author}[1]{\authortrue\def\@author{#1}}

% TikZ predefined drawings
\tikzset{
    antlogocompass/.pic={
        % (-0.866,-0.866) to (0.866,0.866) with 0.245 spikes
        \node[transform shape, scale=0.15, opacity=0.03, above]
            at (0.433,0) {DW~\@version};
        \foreach \phi in {0,90,180,270}{
            \fill[rotate=\phi]
                (-0.034,0) -- (-0.01,0.024) -- (-0.01,1.111) -- (0,1.111) --
                (0.054,0.864) arc (86.41:3.59:0.866) -- (0.687,0.093)
                arc (7.74:82.26:0.693) -- (0.198,0.212) -- (0.444,0.458) --
                (0.451,0.451) -- (0.342,0.169) -- (0.200,0.200) -- (0,0);
        }},
    antlogoant/.pic={
        % (-0.866,-0.334) to (0.866,0.334)
        \fill[even odd rule] % A
            (-0.866,-0.334) -- (-0.668,0.334) -- (-0.473,0.334) --
            (-0.275,-0.334) -- (-0.450,-0.334) -- (-0.476,-0.231) --
            (-0.665,-0.231) -- (-0.691,-0.334)
            (-0.571,0.164) -- (-0.506,-0.092) -- (-0.636,-0.092);
        \fill                % N
            (-0.202,-0.334) -- (-0.202,0.334) -- (-0.04,0.334) --
            (0.156,-0.009) -- (0.156,0.334) -- (0.318,0.334) --
            (0.318,-0.334) -- (0.162,-0.334) -- (-0.041,-0.02) --
            (-0.041,-0.334);
        \fill                % T
            (0.56,-0.334) -- (0.56,0.194) -- (0.414,0.194) --
            (0.414,0.334) -- (0.866,0.334) -- (0.866,0.194) --
            (0.72,0.194) -- (0.72,-0.334);},
    antlogocenter/.pic={
        % (-0.866,-0.177) to (0.866,0.177)
        \fill                % C
            (-0.655,0.085) -- (-0.608,0.106)
            .. controls (-0.625,0.146) and (-0.657,0.184) .. (-0.721,0.184)
            .. controls (-0.8,0.184) and (-0.866,0.126) .. (-0.866,0.001)
            .. controls (-0.866,-0.124) and (-0.8,-0.184) .. (-0.721,-0.184)
            .. controls (-0.657,-0.184) and (-0.625,-0.141) ..
            (-0.608,-0.101) -- (-0.655,-0.08)
            .. controls (-0.66,-0.094) and (-0.68,-0.132) .. (-0.721,-0.132)
            .. controls (-0.78,-0.132) and (-0.807,-0.064) .. (-0.807,0.001)
            .. controls (-0.807,0.066) and (-0.78,0.133) .. (-0.721,0.133)
            .. controls (-0.68,0.133) and (-0.66,0.099) .. (-0.655,0.085);
        \fill                % E
            (-0.534,-0.177) -- (-0.534,0.177) -- (-0.34,0.177) --
            (-0.34,0.125) -- (-0.475,0.125) -- (-0.475,0.029) --
            (-0.343,0.029) -- (-0.343,-0.023) -- (-0.475,-0.023) --
            (-0.475,-0.126) -- (-0.34,-0.126) -- (-0.34,-0.177);
        \fill                % N
            (-0.256,-0.177) -- (-0.256,0.177) -- (-0.197,0.177) --
            (-0.068,-0.055) -- (-0.068,0.177) -- (-0.009,0.177) --
            (-0.009,-0.177) -- (-0.066,-0.177) -- (-0.198,0.062) --
            (-0.198,-0.177);
        \fill                % T
            (0.146,-0.177) -- (0.146,0.125) -- (0.064,0.125) --
            (0.064,0.177) -- (0.285,0.177) -- (0.285,0.125) --
            (0.203,0.125) -- (0.203,-0.177);
        \fill                % E
            (0.36,-0.177) -- (0.36,0.177) -- (0.554,0.177) --
            (0.554,0.125) -- (0.419,0.125) -- (0.419,0.029) --
            (0.551,0.029) -- (0.551,-0.023) -- (0.419,-0.023) --
            (0.419,-0.126) -- (0.554,-0.126) -- (0.554,-0.177);
        \fill[even odd rule] % R
            (0.637,-0.177) -- (0.637,0.177) -- (0.77,0.177)
            .. controls (0.805,0.177) and (0.863,0.146) .. (0.863,0.062)
            .. controls (0.863,0.011) and (0.830,-0.024) .. (0.795,-0.034) --
            (0.866,-0.177) -- (0.799,-0.177) -- (0.739,-0.042) --
            (0.694,-0.042) -- (0.694,-0.177) -- (0.637,-0.177)
            (0.694,0.008) -- (0.694,0.125) -- (0.76,0.125)
            .. controls (0.78,0.125) and (0.803,0.106) .. (0.803,0.062)
            .. controls (0.803,0.036) and (0.78,0.008) .. (0.75,0.008);},
    titlepage/.pic={
        % Background
        \fill[colorB] (-\@xw,-4.5) rectangle (-\@xw+3cm,4.5);
        \fill[colorC] (-\@xw,3.0) -- ({-\@xw+0.75cm},3.75) --
            ({-\@xw+3cm},1.5) -- ({-\@xw+3cm},-3.0) --
            ({-\@xw+2.25cm},-3.75) -- (-\@xw,-1.5);
        \fill[colorA] (-\@xw,4.5) -- ({-\@xw+1.5cm},4.5) --
            ({-\@xw+0.75cm},3.75) ({-\@xw+3cm},-4.5) --
            ({-\@xw+1.5cm},-4.5) -- ({-\@xw+2.25cm},-3.75cm);
        \fill[colorD] ({-\@xw+1.5cm},3.75) -- ({-\@xw+2.25cm},4.5) --
            ({-\@xw+3cm},4.5) -- ({-\@xw+3cm},2.25)
            ({-\@xw+1.5cm},-3.75cm) --({-\@xw+0.75cm},-4.5) --
            (-\@xw,-4.5) -- (-\@xw,-2.25cm);
        % Logo
        \begin{scope}[xshift={-\@xw+1.5cm}]
            \path (0,0.878) pic[color=colorH] {antlogocompass};
            \path (0,-0.811) pic[color=colorH] {antlogoant};
            \path (0,-1.567) pic[color=colorH] {antlogocenter};
        \end{scope}
        % Text widths
        \newlength{\@titlewidth}
        \setlength{\@titlewidth}{2\@xw}
        \addtolength{\@titlewidth}{-4.5cm}
        % Titles and authors
        \node[right, inner sep=0] at ({3.5cm-\@xw},0)
            {\parbox[c][7cm][c]{\@titlewidth}{%
                \raggedright
                \ifpretitle
                    \normalsize\color{black}{\@pretitle} \\[1.5ex]
                \fi
                \iftitle
                    \Large\color{black}{\bfseries{\@title}} \\[1.5ex]
                \fi
                \ifsubtitle
                    \large\color{colorH}{\bfseries{\@subtitle}}%
                    \\[2ex]
                \fi
                \ifauthor
                    \normalsize\color{black}{\@author}
                \fi
            }};},
    fullColorBkgnd/.pic={
        \ifstandard
            \fill[colorB] (-6,-4.5) rectangle (6,4.5);
            \fill[colorC]
                (-6,3.0) -- (-5.25,3.75) -- (-0.75,-0.75) --
                (-3.75,-3.75) -- (-6,-1.5)
                (6,-3.0) -- (5.25,-3.75) -- (0.75,0.75) --
                (3.75,3.75) -- (6,1.5);
            \fill[colorA]
                (-6,4.5) -- (-4.5,4.5) -- (-5.25,3.75)
                (-3,-4.5) -- (-4.5,-4.5) -- (-3.75,-3.75)
                (6,-4.5) -- (4.5,-4.5) -- (5.25,-3.75)
                (3,4.5) -- (4.5,4.5) -- (3.75,3.75);
            \fill[colorD]
                (-4.5,3.75) -- (-3.75,4.5) -- (2.25,4.5) --
                (3,3.75) -- (-0.75,0)
                (4.5,-3.75) -- (3.75,-4.5) -- (-2.25,-4.5) --
                (-3,-3.75) -- (0.75,0)
                (-4.5,-3.75) -- (-5.25,-4.5) -- (-6,-4.5) -- (-6,-2.25)
                (4.5,3.75) --(5.25,4.5) -- (6,4.5) -- (6,2.25);
        \else
            \fill[colorB] (-8,-4.5) rectangle (8,4.5);
            \fill[colorC]
                (-8,3.0) -- (-7.25,3.75) -- (-2.75,-0.75) --
                (-5.75,-3.75) -- (-8,-1.5)
                (8,-3.0) -- (7.25,-3.75) -- (2.75,0.75) --
                (5.75,3.75) -- (8,1.5);
            \fill[colorA]
                (-8,4.5) -- (-6.5,4.5) -- (-7.25,3.75)
                (-5,-4.5) -- (-6.5,-4.5) -- (-5.75,-3.75)
                (8,-4.5) -- (6.5,-4.5) -- (7.25,-3.75)
                (5,4.5) -- (6.5,4.5) -- (5.75,3.75)
                (-2.75,-0.75) -- (0.75,2.75) --
                (2.75,0.75) -- (-0.75,-2.75);
            \fill[colorD]
                (-6.5,3.75) -- (-5.75,4.5) -- (1.75,4.5) -- (-2.75,0)
                (-6.5,-3.75) -- (-7.25,-4.5) -- (-8,-4.5) -- (-8,-2.25)
                (6.5,-3.75) -- (5.75,-4.5) -- (-1.75,-4.5) -- (2.75,0)
                (6.5,3.75) --(7.25,4.5) -- (8,4.5) -- (8,2.25);
        \fi},
    chapterpage/.pic={
        % Background
        \path (0,0) pic{fullColorBkgnd};
        % Text
        \node[inner sep=0, text width={1.53\@xw},
                white, align=flush center]
            at (0,0) {\huge\bfseries\strut\@headername\strut};},
    sectionpage/.pic={
        % Background
        \fill[colorB] (-\@xw,-4.5) rectangle ({-\@xw+1cm},4.5);
        \fill[colorC] (-\@xw,3) -- ({-\@xw+0.75cm},3.75) --
            ({-\@xw+1cm},3.5) -- ({-\@xw+1cm},-2.5) -- (-\@xw,-1.5);
        \fill[colorA] (-\@xw,4.5) -- ({-\@xw+1cm},4.5) --
            ({-\@xw+1cm},4) -- ({-\@xw+0.75cm},3.75);
        \fill[colorD] (-\@xw,-4.5) -- (-\@xw,-2.25) --
            ({-\@xw+1cm},-3.25) -- ({-\@xw+1cm},-4.25) --
            ({-\@xw+0.75cm},-4.5);
        % Title
        \node[right, inner sep=0, text depth=0]
            at ({-\@xw+1.5cm},3.723)
            {\begin{adjustbox}{max width={2\@xw-2.5cm}}
                    \Large\bfseries\ignorespaces\@headername\strut
                \end{adjustbox}};
        \phantomsection\addcontentsline{toc}{section}{\@altheadername}
        % Page number
        \node[inner sep=0, text depth=0, colorH]
            at ({-\@xw+0.5cm},-4) {\bfseries\thepage\strut};},
    closingpage/.pic={
        % Background
        \path (0,0) pic{fullColorBkgnd};
        % Logo
        \path (0,0) pic[scale=2, colorH] {antlogocompass};}}

% Headers and footers
\newcommand{\@headername}{}
\pagestyle{fancy}
\fancyhead[L,R]{}
\fancyhead[C]{\tikz[overlay]{\path (\@xHead,\@yHead) pic{sectionpage};}}
\fancyfoot[L,C,R]{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Title page
\renewcommand{\maketitle}{
    \clearpage\thispagestyle{empty}\justifying\onecolumn%
    \tikz[overlay, xshift=\@xShift, yshift=\@yShift]{
        \path (0,0) pic{titlepage};}
    \clearpage}

% Chapter page
\newcommand{\chapter}[2][]{%
    \clearpage\onecolumn\justifying\thispagestyle{empty}\normalsize%
    \renewcommand{\@headername}{#2}%
    \ifx&#1& % if parameter 1 is empty
        \def\@altheadername{#2}
    \else % if parameter 1 is not empty
        \def\@altheadername{#1}
    \fi
    \phantomsection\addcontentsline{toc}{chapter}{\@altheadername}
    \tikz[overlay, xshift=\@xShift, yshift=\@yShift]{
        \path (0,0) pic{chapterpage};}%
    \clearpage}

% Header name and sections
\renewcommand{\section}[2][]{%
    \clearpage\justifying\onecolumn\normalsize%
    \renewcommand{\@headername}{#2}%
    \ifx&#1& % if parameter 1 is empty
        \def\@altheadername{#2}
    \else % if parameter 1 is not empty
        \def\@altheadername{#1}
    \fi}

% Other sections
\renewcommand{\subsection}[1]{\par\textcolor{colorH}{\textbf{#1}}\par}
\renewcommand{\subsubsection}[1]{\par\textbf{#1}\par}
\renewcommand{\paragraph}[1]{\par\textit{#1}\par}
\renewcommand{\subparagraph}[1]{\par\textit{#1}\par}

% Blank page for arbitrarily-placed graphics and text.
\newcommand\blankpage{\clearpage\null\thispagestyle{empty}}

% Closing page
\newcommand{\closing}{
    \clearpage\thispagestyle{empty}\justifying\onecolumn\normalsize%
    \tikz[overlay, xshift=\@xShift, yshift=\@yShift]{
        \path (0,0) pic{closingpage};}%
    \clearpage}
