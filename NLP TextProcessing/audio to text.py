<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<!-- saved from url=(0046)https://docs.python.org/2/library/os.path.html -->
<html xmlns="http://www.w3.org/1999/xhtml"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    
    
    <title>10.1. os.path — Common pathname manipulations — Python 2.7.13 documentation</title>
    
    <link rel="stylesheet" href="./audio to text_files/classic.css" type="text/css">
    <link rel="stylesheet" href="./audio to text_files/pygments.css" type="text/css">
    
    <script type="text/javascript">
      var DOCUMENTATION_OPTIONS = {
        URL_ROOT:    '../',
        VERSION:     '2.7.13',
        COLLAPSE_INDEX: false,
        FILE_SUFFIX: '.html',
        HAS_SOURCE:  true,
        SOURCELINK_SUFFIX: '.txt'
      };
    </script>
    <script type="text/javascript" src="./audio to text_files/jquery.js.download"></script>
    <script type="text/javascript" src="./audio to text_files/underscore.js.download"></script>
    <script type="text/javascript" src="./audio to text_files/doctools.js.download"></script>
    <script type="text/javascript" src="./audio to text_files/sidebar.js.download"></script>
    <link rel="search" type="application/opensearchdescription+xml" title="Search within Python 2.7.13 documentation" href="https://docs.python.org/2/_static/opensearch.xml">
    <link rel="author" title="About these documents" href="https://docs.python.org/2/about.html">
    <link rel="index" title="Index" href="https://docs.python.org/2/genindex.html">
    <link rel="search" title="Search" href="https://docs.python.org/2/search.html">
    <link rel="copyright" title="Copyright" href="https://docs.python.org/2/copyright.html">
    <link rel="next" title="10.2. fileinput — Iterate over lines from multiple input streams" href="https://docs.python.org/2/library/fileinput.html">
    <link rel="prev" title="10. File and Directory Access" href="https://docs.python.org/2/library/filesys.html">
    <link rel="shortcut icon" type="image/png" href="./audio to text_files/py.png">
    <link rel="canonical" href="https://docs.python.org/2/library/os.path.html">
    <script type="text/javascript" src="./audio to text_files/copybutton.js.download"></script>
    <script type="text/javascript" src="./audio to text_files/switchers.js.download"></script>
 
    

  </head>
  <body cz-shortcut-listen="true">  
    <div class="related" role="navigation" aria-label="related navigation">
      <h3>Navigation</h3>
      <ul>
        <li class="right" style="margin-right: 10px">
          <a href="https://docs.python.org/2/genindex.html" title="General Index" accesskey="I">index</a></li>
        <li class="right">
          <a href="https://docs.python.org/2/py-modindex.html" title="Python Module Index">modules</a> |</li>
        <li class="right">
          <a href="https://docs.python.org/2/library/fileinput.html" title="10.2. fileinput — Iterate over lines from multiple input streams" accesskey="N">next</a> |</li>
        <li class="right">
          <a href="https://docs.python.org/2/library/filesys.html" title="10. File and Directory Access" accesskey="P">previous</a> |</li>
        <li><img src="./audio to text_files/py.png" alt="" style="vertical-align: middle; margin-top: -1px"></li>
        <li><a href="https://www.python.org/">Python</a> »</li>
        <li>
          <span class="language_switcher_placeholder"><select><option value="en" selected="selected">English</option><option value="fr">French</option><option value="ja">Japanese</option></select></span>
          <span class="version_switcher_placeholder"><select><option value="3.7">dev (3.7)</option><option value="3.6">3.6</option><option value="3.5">3.5</option><option value="3.4">3.4</option><option value="3.3">3.3</option><option value="2.7" selected="selected">2.7.13</option></select></span>
          <a href="https://docs.python.org/2/index.html">Documentation</a> »
        </li>

          <li class="nav-item nav-item-1"><a href="https://docs.python.org/2/library/index.html">The Python Standard Library</a> »</li>
          <li class="nav-item nav-item-2"><a href="https://docs.python.org/2/library/filesys.html" accesskey="U">10. File and Directory Access</a> »</li> 
      </ul>
    </div>    

    <div class="document">
      <div class="documentwrapper">
        <div class="bodywrapper">
          <div class="body" role="main">
            
  <div class="section" id="module-os.path">
<span id="os-path-common-pathname-manipulations"></span><h1>10.1. <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#module-os.path" title="os.path: Operations on pathnames."><code class="xref py py-mod docutils literal"><span class="pre">os.path</span></code></a> — Common pathname manipulations<a class="headerlink" href="https://docs.python.org/2/library/os.path.html#module-os.path" title="Permalink to this headline">¶</a></h1>
<p id="index-0">This module implements some useful functions on pathnames. To read or
write files see <a class="reference internal" href="https://docs.python.org/2/library/functions.html#open" title="open"><code class="xref py py-func docutils literal"><span class="pre">open()</span></code></a>, and for accessing the filesystem see the
<a class="reference internal" href="https://docs.python.org/2/library/os.html#module-os" title="os: Miscellaneous operating system interfaces."><code class="xref py py-mod docutils literal"><span class="pre">os</span></code></a> module.</p>
<div class="admonition note">
<p class="first admonition-title">Note</p>
<p class="last">On Windows, many of these functions do not properly support UNC pathnames.
<a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.splitunc" title="os.path.splitunc"><code class="xref py py-func docutils literal"><span class="pre">splitunc()</span></code></a> and <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.ismount" title="os.path.ismount"><code class="xref py py-func docutils literal"><span class="pre">ismount()</span></code></a> do handle them correctly.</p>
</div>
<p>Unlike a unix shell, Python does not do any <em>automatic</em> path expansions.
Functions such as <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.expanduser" title="os.path.expanduser"><code class="xref py py-func docutils literal"><span class="pre">expanduser()</span></code></a> and <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.expandvars" title="os.path.expandvars"><code class="xref py py-func docutils literal"><span class="pre">expandvars()</span></code></a> can be invoked
explicitly when an application desires shell-like path expansion.  (See also
the <a class="reference internal" href="https://docs.python.org/2/library/glob.html#module-glob" title="glob: Unix shell style pathname pattern expansion."><code class="xref py py-mod docutils literal"><span class="pre">glob</span></code></a> module.)</p>
<div class="admonition note">
<p class="first admonition-title">Note</p>
<p>Since different operating systems have different path name conventions, there
are several versions of this module in the standard library.  The
<a class="reference internal" href="https://docs.python.org/2/library/os.path.html#module-os.path" title="os.path: Operations on pathnames."><code class="xref py py-mod docutils literal"><span class="pre">os.path</span></code></a> module is always the path module suitable for the operating
system Python is running on, and therefore usable for local paths.  However,
you can also import and use the individual modules if you want to manipulate
a path that is <em>always</em> in one of the different formats.  They all have the
same interface:</p>
<ul class="last simple">
<li><code class="xref py py-mod docutils literal"><span class="pre">posixpath</span></code> for UNIX-style paths</li>
<li><code class="xref py py-mod docutils literal"><span class="pre">ntpath</span></code> for Windows paths</li>
<li><a class="reference internal" href="https://docs.python.org/2/library/macpath.html#module-macpath" title="macpath: Mac OS 9 path manipulation functions."><code class="xref py py-mod docutils literal"><span class="pre">macpath</span></code></a> for old-style MacOS paths</li>
<li><code class="xref py py-mod docutils literal"><span class="pre">os2emxpath</span></code> for OS/2 EMX paths</li>
</ul>
</div>
<dl class="function">
<dt id="os.path.abspath">
<code class="descclassname">os.path.</code><code class="descname">abspath</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.abspath" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a normalized absolutized version of the pathname <em>path</em>. On most
platforms, this is equivalent to calling the function <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.normpath" title="os.path.normpath"><code class="xref py py-func docutils literal"><span class="pre">normpath()</span></code></a> as
follows: <code class="docutils literal"><span class="pre">normpath(join(os.getcwd(),</span> <span class="pre">path))</span></code>.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 1.5.2.</span></p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.basename">
<code class="descclassname">os.path.</code><code class="descname">basename</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.basename" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the base name of pathname <em>path</em>.  This is the second element of the
pair returned by passing <em>path</em> to the function <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.split" title="os.path.split"><code class="xref py py-func docutils literal"><span class="pre">split()</span></code></a>.  Note that
the result of this function is different
from the Unix <strong class="program">basename</strong> program; where <strong class="program">basename</strong> for
<code class="docutils literal"><span class="pre">'/foo/bar/'</span></code> returns <code class="docutils literal"><span class="pre">'bar'</span></code>, the <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.basename" title="os.path.basename"><code class="xref py py-func docutils literal"><span class="pre">basename()</span></code></a> function returns an
empty string (<code class="docutils literal"><span class="pre">''</span></code>).</p>
</dd></dl>

<dl class="function">
<dt id="os.path.commonprefix">
<code class="descclassname">os.path.</code><code class="descname">commonprefix</code><span class="sig-paren">(</span><em>list</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.commonprefix" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the longest path prefix (taken character-by-character) that is a prefix
of all paths in  <em>list</em>.  If <em>list</em> is empty, return the empty string (<code class="docutils literal"><span class="pre">''</span></code>).
Note that this may return invalid paths because it works a character at a time.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.dirname">
<code class="descclassname">os.path.</code><code class="descname">dirname</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.dirname" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the directory name of pathname <em>path</em>.  This is the first element of
the pair returned by passing <em>path</em> to the function <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.split" title="os.path.split"><code class="xref py py-func docutils literal"><span class="pre">split()</span></code></a>.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.exists">
<code class="descclassname">os.path.</code><code class="descname">exists</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.exists" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if <em>path</em> refers to an existing path.  Returns <code class="docutils literal"><span class="pre">False</span></code> for
broken symbolic links. On some platforms, this function may return <code class="docutils literal"><span class="pre">False</span></code> if
permission is not granted to execute <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.stat" title="os.stat"><code class="xref py py-func docutils literal"><span class="pre">os.stat()</span></code></a> on the requested file, even
if the <em>path</em> physically exists.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.lexists">
<code class="descclassname">os.path.</code><code class="descname">lexists</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.lexists" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if <em>path</em> refers to an existing path. Returns <code class="docutils literal"><span class="pre">True</span></code> for
broken symbolic links.   Equivalent to <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.exists" title="os.path.exists"><code class="xref py py-func docutils literal"><span class="pre">exists()</span></code></a> on platforms lacking
<a class="reference internal" href="https://docs.python.org/2/library/os.html#os.lstat" title="os.lstat"><code class="xref py py-func docutils literal"><span class="pre">os.lstat()</span></code></a>.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 2.4.</span></p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.expanduser">
<code class="descclassname">os.path.</code><code class="descname">expanduser</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.expanduser" title="Permalink to this definition">¶</a></dt>
<dd><p>On Unix and Windows, return the argument with an initial component of <code class="docutils literal"><span class="pre">~</span></code> or
<code class="docutils literal"><span class="pre">~user</span></code> replaced by that <em>user</em>’s home directory.</p>
<p id="index-1">On Unix, an initial <code class="docutils literal"><span class="pre">~</span></code> is replaced by the environment variable <span class="target" id="index-2"></span><code class="xref std std-envvar docutils literal"><span class="pre">HOME</span></code>
if it is set; otherwise the current user’s home directory is looked up in the
password directory through the built-in module <a class="reference internal" href="https://docs.python.org/2/library/pwd.html#module-pwd" title="pwd: The password database (getpwnam() and friends). (Unix)"><code class="xref py py-mod docutils literal"><span class="pre">pwd</span></code></a>. An initial <code class="docutils literal"><span class="pre">~user</span></code>
is looked up directly in the password directory.</p>
<p>On Windows, <span class="target" id="index-3"></span><code class="xref std std-envvar docutils literal"><span class="pre">HOME</span></code> and <span class="target" id="index-4"></span><code class="xref std std-envvar docutils literal"><span class="pre">USERPROFILE</span></code> will be used if set,
otherwise a combination of <span class="target" id="index-5"></span><code class="xref std std-envvar docutils literal"><span class="pre">HOMEPATH</span></code> and <span class="target" id="index-6"></span><code class="xref std std-envvar docutils literal"><span class="pre">HOMEDRIVE</span></code> will be
used.  An initial <code class="docutils literal"><span class="pre">~user</span></code> is handled by stripping the last directory component
from the created user path derived above.</p>
<p>If the expansion fails or if the path does not begin with a tilde, the path is
returned unchanged.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.expandvars">
<code class="descclassname">os.path.</code><code class="descname">expandvars</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.expandvars" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the argument with environment variables expanded.  Substrings of the form
<code class="docutils literal"><span class="pre">$name</span></code> or <code class="docutils literal"><span class="pre">${name}</span></code> are replaced by the value of environment variable
<em>name</em>.  Malformed variable names and references to non-existing variables are
left unchanged.</p>
<p>On Windows, <code class="docutils literal"><span class="pre">%name%</span></code> expansions are supported in addition to <code class="docutils literal"><span class="pre">$name</span></code> and
<code class="docutils literal"><span class="pre">${name}</span></code>.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.getatime">
<code class="descclassname">os.path.</code><code class="descname">getatime</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.getatime" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the time of last access of <em>path</em>.  The return value is a number giving
the number of seconds since the epoch (see the  <a class="reference internal" href="https://docs.python.org/2/library/time.html#module-time" title="time: Time access and conversions."><code class="xref py py-mod docutils literal"><span class="pre">time</span></code></a> module).  Raise
<a class="reference internal" href="https://docs.python.org/2/library/os.html#os.error" title="os.error"><code class="xref py py-exc docutils literal"><span class="pre">os.error</span></code></a> if the file does not exist or is inaccessible.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 1.5.2.</span></p>
</div>
<div class="versionchanged">
<p><span class="versionmodified">Changed in version 2.3: </span>If <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.stat_float_times" title="os.stat_float_times"><code class="xref py py-func docutils literal"><span class="pre">os.stat_float_times()</span></code></a> returns <code class="docutils literal"><span class="pre">True</span></code>, the result is a floating point
number.</p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.getmtime">
<code class="descclassname">os.path.</code><code class="descname">getmtime</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.getmtime" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the time of last modification of <em>path</em>.  The return value is a number
giving the number of seconds since the epoch (see the  <a class="reference internal" href="https://docs.python.org/2/library/time.html#module-time" title="time: Time access and conversions."><code class="xref py py-mod docutils literal"><span class="pre">time</span></code></a> module).
Raise <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.error" title="os.error"><code class="xref py py-exc docutils literal"><span class="pre">os.error</span></code></a> if the file does not exist or is inaccessible.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 1.5.2.</span></p>
</div>
<div class="versionchanged">
<p><span class="versionmodified">Changed in version 2.3: </span>If <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.stat_float_times" title="os.stat_float_times"><code class="xref py py-func docutils literal"><span class="pre">os.stat_float_times()</span></code></a> returns <code class="docutils literal"><span class="pre">True</span></code>, the result is a floating point
number.</p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.getctime">
<code class="descclassname">os.path.</code><code class="descname">getctime</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.getctime" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the system’s ctime which, on some systems (like Unix) is the time of the
last metadata change, and, on others (like Windows), is the creation time for <em>path</em>.
The return value is a number giving the number of seconds since the epoch (see
the  <a class="reference internal" href="https://docs.python.org/2/library/time.html#module-time" title="time: Time access and conversions."><code class="xref py py-mod docutils literal"><span class="pre">time</span></code></a> module).  Raise <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.error" title="os.error"><code class="xref py py-exc docutils literal"><span class="pre">os.error</span></code></a> if the file does not exist or
is inaccessible.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 2.3.</span></p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.getsize">
<code class="descclassname">os.path.</code><code class="descname">getsize</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.getsize" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the size, in bytes, of <em>path</em>.  Raise <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.error" title="os.error"><code class="xref py py-exc docutils literal"><span class="pre">os.error</span></code></a> if the file does
not exist or is inaccessible.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 1.5.2.</span></p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.isabs">
<code class="descclassname">os.path.</code><code class="descname">isabs</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.isabs" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if <em>path</em> is an absolute pathname.  On Unix, that means it
begins with a slash, on Windows that it begins with a (back)slash after chopping
off a potential drive letter.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.isfile">
<code class="descclassname">os.path.</code><code class="descname">isfile</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.isfile" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if <em>path</em> is an existing regular file.  This follows symbolic
links, so both <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.islink" title="os.path.islink"><code class="xref py py-func docutils literal"><span class="pre">islink()</span></code></a> and <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.isfile" title="os.path.isfile"><code class="xref py py-func docutils literal"><span class="pre">isfile()</span></code></a> can be true for the same path.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.isdir">
<code class="descclassname">os.path.</code><code class="descname">isdir</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.isdir" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if <em>path</em> is an existing directory.  This follows symbolic
links, so both <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.islink" title="os.path.islink"><code class="xref py py-func docutils literal"><span class="pre">islink()</span></code></a> and <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.isdir" title="os.path.isdir"><code class="xref py py-func docutils literal"><span class="pre">isdir()</span></code></a> can be true for the same path.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.islink">
<code class="descclassname">os.path.</code><code class="descname">islink</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.islink" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if <em>path</em> refers to a directory entry that is a symbolic link.
Always <code class="docutils literal"><span class="pre">False</span></code> if symbolic links are not supported by the Python runtime.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.ismount">
<code class="descclassname">os.path.</code><code class="descname">ismount</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.ismount" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if pathname <em>path</em> is a <em class="dfn">mount point</em>: a point in a file
system where a different file system has been mounted.  The function checks
whether <em>path</em>’s parent, <code class="file docutils literal"><span class="pre">path/..</span></code>, is on a different device than <em>path</em>,
or whether <code class="file docutils literal"><span class="pre">path/..</span></code> and <em>path</em> point to the same i-node on the same
device — this should detect mount points for all Unix and POSIX variants.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.join">
<code class="descclassname">os.path.</code><code class="descname">join</code><span class="sig-paren">(</span><em>path</em>, <em>*paths</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.join" title="Permalink to this definition">¶</a></dt>
<dd><p>Join one or more path components intelligently.  The return value is the
concatenation of <em>path</em> and any members of <em>*paths</em> with exactly one
directory separator (<code class="docutils literal"><span class="pre">os.sep</span></code>) following each non-empty part except the
last, meaning that the result will only end in a separator if the last
part is empty.  If a component is an absolute path, all previous
components are thrown away and joining continues from the absolute path
component.</p>
<p>On Windows, the drive letter is not reset when an absolute path component
(e.g., <code class="docutils literal"><span class="pre">r'\foo'</span></code>) is encountered.  If a component contains a drive
letter, all previous components are thrown away and the drive letter is
reset.  Note that since there is a current directory for each drive,
<code class="docutils literal"><span class="pre">os.path.join("c:",</span> <span class="pre">"foo")</span></code> represents a path relative to the current
directory on drive <code class="file docutils literal"><span class="pre">C:</span></code> (<code class="file docutils literal"><span class="pre">c:foo</span></code>), not <code class="file docutils literal"><span class="pre">c:\foo</span></code>.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.normcase">
<code class="descclassname">os.path.</code><code class="descname">normcase</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.normcase" title="Permalink to this definition">¶</a></dt>
<dd><p>Normalize the case of a pathname.  On Unix and Mac OS X, this returns the
path unchanged; on case-insensitive filesystems, it converts the path to
lowercase.  On Windows, it also converts forward slashes to backward slashes.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.normpath">
<code class="descclassname">os.path.</code><code class="descname">normpath</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.normpath" title="Permalink to this definition">¶</a></dt>
<dd><p>Normalize a pathname by collapsing redundant separators and up-level
references so that <code class="docutils literal"><span class="pre">A//B</span></code>, <code class="docutils literal"><span class="pre">A/B/</span></code>, <code class="docutils literal"><span class="pre">A/./B</span></code> and <code class="docutils literal"><span class="pre">A/foo/../B</span></code> all
become <code class="docutils literal"><span class="pre">A/B</span></code>.  This string manipulation may change the meaning of a path
that contains symbolic links.  On Windows, it converts forward slashes to
backward slashes. To normalize case, use <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.normcase" title="os.path.normcase"><code class="xref py py-func docutils literal"><span class="pre">normcase()</span></code></a>.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.realpath">
<code class="descclassname">os.path.</code><code class="descname">realpath</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.realpath" title="Permalink to this definition">¶</a></dt>
<dd><p>Return the canonical path of the specified filename, eliminating any symbolic
links encountered in the path (if they are supported by the operating system).</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 2.2.</span></p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.relpath">
<code class="descclassname">os.path.</code><code class="descname">relpath</code><span class="sig-paren">(</span><em>path</em><span class="optional">[</span>, <em>start</em><span class="optional">]</span><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.relpath" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a relative filepath to <em>path</em> either from the current directory or
from an optional <em>start</em> directory.  This is a path computation:  the
filesystem is not accessed to confirm the existence or nature of <em>path</em> or
<em>start</em>.</p>
<p><em>start</em> defaults to <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.curdir" title="os.curdir"><code class="xref py py-attr docutils literal"><span class="pre">os.curdir</span></code></a>.</p>
<p>Availability:  Windows, Unix.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 2.6.</span></p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.samefile">
<code class="descclassname">os.path.</code><code class="descname">samefile</code><span class="sig-paren">(</span><em>path1</em>, <em>path2</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.samefile" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if both pathname arguments refer to the same file or directory
(as indicated by device number and i-node number). Raise an exception if an
<a class="reference internal" href="https://docs.python.org/2/library/os.html#os.stat" title="os.stat"><code class="xref py py-func docutils literal"><span class="pre">os.stat()</span></code></a> call on either pathname fails.</p>
<p>Availability: Unix.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.sameopenfile">
<code class="descclassname">os.path.</code><code class="descname">sameopenfile</code><span class="sig-paren">(</span><em>fp1</em>, <em>fp2</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.sameopenfile" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if the file descriptors <em>fp1</em> and <em>fp2</em> refer to the same file.</p>
<p>Availability: Unix.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.samestat">
<code class="descclassname">os.path.</code><code class="descname">samestat</code><span class="sig-paren">(</span><em>stat1</em>, <em>stat2</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.samestat" title="Permalink to this definition">¶</a></dt>
<dd><p>Return <code class="docutils literal"><span class="pre">True</span></code> if the stat tuples <em>stat1</em> and <em>stat2</em> refer to the same file.
These structures may have been returned by <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.fstat" title="os.fstat"><code class="xref py py-func docutils literal"><span class="pre">os.fstat()</span></code></a>,
<a class="reference internal" href="https://docs.python.org/2/library/os.html#os.lstat" title="os.lstat"><code class="xref py py-func docutils literal"><span class="pre">os.lstat()</span></code></a>, or <a class="reference internal" href="https://docs.python.org/2/library/os.html#os.stat" title="os.stat"><code class="xref py py-func docutils literal"><span class="pre">os.stat()</span></code></a>.  This function implements the
underlying comparison used by <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.samefile" title="os.path.samefile"><code class="xref py py-func docutils literal"><span class="pre">samefile()</span></code></a> and <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.sameopenfile" title="os.path.sameopenfile"><code class="xref py py-func docutils literal"><span class="pre">sameopenfile()</span></code></a>.</p>
<p>Availability: Unix.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.split">
<code class="descclassname">os.path.</code><code class="descname">split</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.split" title="Permalink to this definition">¶</a></dt>
<dd><p>Split the pathname <em>path</em> into a pair, <code class="docutils literal"><span class="pre">(head,</span> <span class="pre">tail)</span></code> where <em>tail</em> is the
last pathname component and <em>head</em> is everything leading up to that.  The
<em>tail</em> part will never contain a slash; if <em>path</em> ends in a slash, <em>tail</em>
will be empty.  If there is no slash in <em>path</em>, <em>head</em> will be empty.  If
<em>path</em> is empty, both <em>head</em> and <em>tail</em> are empty.  Trailing slashes are
stripped from <em>head</em> unless it is the root (one or more slashes only).  In
all cases, <code class="docutils literal"><span class="pre">join(head,</span> <span class="pre">tail)</span></code> returns a path to the same location as <em>path</em>
(but the strings may differ).  Also see the functions <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.dirname" title="os.path.dirname"><code class="xref py py-func docutils literal"><span class="pre">dirname()</span></code></a> and
<a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.basename" title="os.path.basename"><code class="xref py py-func docutils literal"><span class="pre">basename()</span></code></a>.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.splitdrive">
<code class="descclassname">os.path.</code><code class="descname">splitdrive</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.splitdrive" title="Permalink to this definition">¶</a></dt>
<dd><p>Split the pathname <em>path</em> into a pair <code class="docutils literal"><span class="pre">(drive,</span> <span class="pre">tail)</span></code> where <em>drive</em> is either
a drive specification or the empty string.  On systems which do not use drive
specifications, <em>drive</em> will always be the empty string.  In all cases, <code class="docutils literal"><span class="pre">drive</span>
<span class="pre">+</span> <span class="pre">tail</span></code> will be the same as <em>path</em>.</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 1.3.</span></p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.splitext">
<code class="descclassname">os.path.</code><code class="descname">splitext</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.splitext" title="Permalink to this definition">¶</a></dt>
<dd><p>Split the pathname <em>path</em> into a pair <code class="docutils literal"><span class="pre">(root,</span> <span class="pre">ext)</span></code>  such that <code class="docutils literal"><span class="pre">root</span> <span class="pre">+</span> <span class="pre">ext</span> <span class="pre">==</span>
<span class="pre">path</span></code>, and <em>ext</em> is empty or begins with a period and contains at most one
period. Leading periods on the basename are  ignored; <code class="docutils literal"><span class="pre">splitext('.cshrc')</span></code>
returns  <code class="docutils literal"><span class="pre">('.cshrc',</span> <span class="pre">'')</span></code>.</p>
<div class="versionchanged">
<p><span class="versionmodified">Changed in version 2.6: </span>Earlier versions could produce an empty root when the only period was the
first character.</p>
</div>
</dd></dl>

<dl class="function">
<dt id="os.path.splitunc">
<code class="descclassname">os.path.</code><code class="descname">splitunc</code><span class="sig-paren">(</span><em>path</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.splitunc" title="Permalink to this definition">¶</a></dt>
<dd><p>Split the pathname <em>path</em> into a pair <code class="docutils literal"><span class="pre">(unc,</span> <span class="pre">rest)</span></code> so that <em>unc</em> is the UNC
mount point (such as <code class="docutils literal"><span class="pre">r'\\host\mount'</span></code>), if present, and <em>rest</em> the rest of
the path (such as  <code class="docutils literal"><span class="pre">r'\path\file.ext'</span></code>).  For paths containing drive letters,
<em>unc</em> will always be the empty string.</p>
<p>Availability:  Windows.</p>
</dd></dl>

<dl class="function">
<dt id="os.path.walk">
<code class="descclassname">os.path.</code><code class="descname">walk</code><span class="sig-paren">(</span><em>path</em>, <em>visit</em>, <em>arg</em><span class="sig-paren">)</span><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.walk" title="Permalink to this definition">¶</a></dt>
<dd><p>Calls the function <em>visit</em> with arguments <code class="docutils literal"><span class="pre">(arg,</span> <span class="pre">dirname,</span> <span class="pre">names)</span></code> for each
directory in the directory tree rooted at <em>path</em> (including <em>path</em> itself, if it
is a directory).  The argument <em>dirname</em> specifies the visited directory, the
argument <em>names</em> lists the files in the directory (gotten from
<code class="docutils literal"><span class="pre">os.listdir(dirname)</span></code>). The <em>visit</em> function may modify <em>names</em> to influence
the set of directories visited below <em>dirname</em>, e.g. to avoid visiting certain
parts of the tree.  (The object referred to by <em>names</em> must be modified in
place, using <a class="reference internal" href="https://docs.python.org/2/reference/simple_stmts.html#del"><code class="xref std std-keyword docutils literal"><span class="pre">del</span></code></a> or slice assignment.)</p>
<div class="admonition note">
<p class="first admonition-title">Note</p>
<p class="last">Symbolic links to directories are not treated as subdirectories, and that
<a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.walk" title="os.path.walk"><code class="xref py py-func docutils literal"><span class="pre">walk()</span></code></a> therefore will not visit them. To visit linked directories you must
identify them with <code class="docutils literal"><span class="pre">os.path.islink(file)</span></code> and <code class="docutils literal"><span class="pre">os.path.isdir(file)</span></code>, and
invoke <a class="reference internal" href="https://docs.python.org/2/library/os.path.html#os.path.walk" title="os.path.walk"><code class="xref py py-func docutils literal"><span class="pre">walk()</span></code></a> as necessary.</p>
</div>
<div class="admonition note">
<p class="first admonition-title">Note</p>
<p class="last">This function is deprecated and has been removed in Python 3 in favor of
<a class="reference internal" href="https://docs.python.org/2/library/os.html#os.walk" title="os.walk"><code class="xref py py-func docutils literal"><span class="pre">os.walk()</span></code></a>.</p>
</div>
</dd></dl>

<dl class="data">
<dt id="os.path.supports_unicode_filenames">
<code class="descclassname">os.path.</code><code class="descname">supports_unicode_filenames</code><a class="headerlink" href="https://docs.python.org/2/library/os.path.html#os.path.supports_unicode_filenames" title="Permalink to this definition">¶</a></dt>
<dd><p><code class="docutils literal"><span class="pre">True</span></code> if arbitrary Unicode strings can be used as file names (within limitations
imposed by the file system).</p>
<div class="versionadded">
<p><span class="versionmodified">New in version 2.3.</span></p>
</div>
</dd></dl>

</div>


          </div>
        </div>
      </div>
      <div class="sphinxsidebar" role="navigation" aria-label="main navigation">
        <div class="sphinxsidebarwrapper" style="float: left; margin-right: 0px; width: 202px;">
  <h4>Previous topic</h4>
  <p class="topless"><a href="https://docs.python.org/2/library/filesys.html" title="previous chapter">10. File and Directory Access</a></p>
  <h4>Next topic</h4>
  <p class="topless"><a href="https://docs.python.org/2/library/fileinput.html" title="next chapter">10.2. <code class="docutils literal"><span class="pre">fileinput</span></code> — Iterate over lines from multiple input streams</a></p>
<h3>This Page</h3>
<ul class="this-page-menu">
  <li><a href="https://docs.python.org/2/bugs.html">Report a Bug</a></li>
  <li><a href="https://github.com/python/cpython/blob/2.7/Doc/library/os.path.rst" rel="nofollow">Show Source</a>
  </li>
</ul>

<div id="searchbox" style="" role="search">
  <h3>Quick search</h3>
    <form class="search" action="https://docs.python.org/2/search.html" method="get">
      <div><input type="text" name="q"></div>
      <div><input type="submit" value="Go"></div>
      <input type="hidden" name="check_keywords" value="yes">
      <input type="hidden" name="area" value="default">
    </form>
</div>
<script type="text/javascript">$('#searchbox').show(0);</script>
        </div>
      <div id="sidebarbutton" title="Collapse sidebar" style="color: rgb(255, 255, 255); border-left: 1px solid rgb(19, 63, 82); font-size: 1.2em; cursor: pointer; height: 3577px; padding-top: 1px; margin-left: 218px;"><span style="display: block; margin-top: 522px;">«</span></div></div>
      <div class="clearer"></div>
    </div>  
    <div class="related" role="navigation" aria-label="related navigation">
      <h3>Navigation</h3>
      <ul>
        <li class="right" style="margin-right: 10px">
          <a href="https://docs.python.org/2/genindex.html" title="General Index">index</a></li>
        <li class="right">
          <a href="https://docs.python.org/2/py-modindex.html" title="Python Module Index">modules</a> |</li>
        <li class="right">
          <a href="https://docs.python.org/2/library/fileinput.html" title="10.2. fileinput — Iterate over lines from multiple input streams">next</a> |</li>
        <li class="right">
          <a href="https://docs.python.org/2/library/filesys.html" title="10. File and Directory Access">previous</a> |</li>
        <li><img src="./audio to text_files/py.png" alt="" style="vertical-align: middle; margin-top: -1px"></li>
        <li><a href="https://www.python.org/">Python</a> »</li>
        <li>
          <span class="language_switcher_placeholder"><select><option value="en" selected="selected">English</option><option value="fr">French</option><option value="ja">Japanese</option></select></span>
          <span class="version_switcher_placeholder"><select><option value="3.7">dev (3.7)</option><option value="3.6">3.6</option><option value="3.5">3.5</option><option value="3.4">3.4</option><option value="3.3">3.3</option><option value="2.7" selected="selected">2.7.13</option></select></span>
          <a href="https://docs.python.org/2/index.html">Documentation</a> »
        </li>

          <li class="nav-item nav-item-1"><a href="https://docs.python.org/2/library/index.html">The Python Standard Library</a> »</li>
          <li class="nav-item nav-item-2"><a href="https://docs.python.org/2/library/filesys.html">10. File and Directory Access</a> »</li> 
      </ul>
    </div>  
    <div class="footer">
    © <a href="https://docs.python.org/2/copyright.html">Copyright</a> 1990-2017, Python Software Foundation.
    <br>
    The Python Software Foundation is a non-profit corporation.
    <a href="https://www.python.org/psf/donations/">Please donate.</a>
    <br>
    Last updated on Aug 22, 2017.
    <a href="https://docs.python.org/2/bugs.html">Found a bug</a>?
    <br>
    Created using <a href="http://sphinx.pocoo.org/">Sphinx</a> 1.6.2.
    </div>

  
</body></html>