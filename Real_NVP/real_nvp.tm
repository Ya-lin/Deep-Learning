<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Real NVP>>

  Let <math|x=<around*|(|x<rsub|1>,x<rsub|2>,x<rsub|3>,x<rsub|4>|)>\<sim\>P<rsub|X>>
  and <math|z=<around*|(|z<rsub|1>,z<rsub|2>,z<rsub|3>,z<rsub|4>|)>\<sim\><with|font|cal|N><around*|(|0,1|)>>
  . The goal is to design an invertible network such that it can transform
  data distribution <math|P<rsub|X>> to the normal distribution
  <math|<with|font|cal|N><around*|(|0,1|)>>.

  Give the boolean vector <math|b=<around*|(|1,1,0,0|)>> and two mappings

  <\eqnarray*>
    <tformat|<table|<row|<cell|s>|<cell|:>|<cell|\<bbb-R\><rsup|4>\<rightarrow\>\<bbb-R\><rsup|4>>>|<row|<cell|t>|<cell|:>|<cell|\<bbb-R\><rsup|4>\<rightarrow\>\<bbb-R\><rsup|4>,>>>>
  </eqnarray*>

  \ the coupling layer <math|z=c<around*|(|x|)>> is defined as the
  transformation

  <\eqnarray*>
    <tformat|<table|<row|<cell|c>|<cell|:>|<cell|\<bbb-R\><rsup|4>\<rightarrow\>\<bbb-R\><rsup|4>>>|<row|<cell|>|<cell|>|<cell|x\<rightarrow\>b\<odot\>x+<around*|(|1-b|)>\<odot\><around*|[|x\<odot\>exp<around*|(|s<around*|(|b\<odot\>x|)>|)>+t<around*|(|b\<odot\>x|)>|]>>>>>
  </eqnarray*>

  where <math|exp> is component-wise, and <math|\<odot\>> is the Hadamard
  component-wise product. Particularly,

  <\eqnarray*>
    <tformat|<table|<row|<cell|z<rsub|1>>|<cell|=>|<cell|x<rsub|1>>>|<row|<cell|z<rsub|2>>|<cell|=>|<cell|x<rsub|2>>>|<row|<cell|z<rsub|3>>|<cell|=>|<cell|x<rsub|3>\<cdot\>exp<around*|(|s<rsub|3><around*|(|b\<odot\>x|)>|)>+t<rsub|3><around*|(|b\<odot\>x|)>>>|<row|<cell|z<rsub|4>>|<cell|=>|<cell|x<rsub|4>\<cdot\>exp<around*|(|s<rsub|4><around*|(|b\<odot\>x|)>|)>+t<rsub|4><around*|(|b\<odot\>x|)>>>>>
  </eqnarray*>

  where <math|><math|s<rsub|i><around*|(|b\<odot\>x|)>> and
  <math|t<rsub|i><around*|(|b\<odot\>x|)>> is the <math|i>-th component of
  <math|s<around*|(|b\<odot\>x|)>> and <math|t<around*|(|b\<odot\>x|)>>,
  respectively. By arranging terms,

  <\eqnarray*>
    <tformat|<table|<row|<cell|x<rsub|1>>|<cell|=>|<cell|z<rsub|1>>>|<row|<cell|x<rsub|2>>|<cell|=>|<cell|z<rsub|2>>>|<row|<cell|x<rsub|3>>|<cell|=>|<cell|<around*|[|z<rsub|3>-t<rsub|3><around*|(|b\<odot\>x|)>|]>\<cdot\>exp<around*|(|-s<rsub|3><around*|(|b\<odot\>x|)>|)>>>|<row|<cell|x<rsub|4>>|<cell|=>|<cell|<around*|[|z<rsub|4>-t<rsub|4><around*|(|b\<odot\>x|)>|]>\<cdot\>exp<around*|(|-s<rsub|4><around*|(|b\<odot\>x|)>|)>>>>>
  </eqnarray*>

  Note that <math|b\<odot\>x=<around*|(|1,1,0,0|)>\<odot\><around*|(|x<rsub|1>,x<rsub|2>,x<rsub|3>,x<rsub|4>|)>=<around*|(|x<rsub|1>,x<rsub|2>,0,0|)>=<around*|(|z<rsub|1>,z<rsub|2>,0,0|)>=b\<odot\>z>,
  we have the inverse mapping <math|x=c<rsup|-1><around*|(|z|)>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|x<rsub|1>>|<cell|=>|<cell|z<rsub|1>>>|<row|<cell|x<rsub|2>>|<cell|=>|<cell|z<rsub|2>>>|<row|<cell|x<rsub|3>>|<cell|=>|<cell|<around*|[|z<rsub|3>-t<rsub|3><around*|(|b\<odot\>z|)>|]>\<cdot\>exp<around*|(|-s<rsub|3><around*|(|b\<odot\>z|)>|)>>>|<row|<cell|x<rsub|4>>|<cell|=>|<cell|<around*|[|z<rsub|4>-t<rsub|4><around*|(|b\<odot\>z|)>|]>\<cdot\>exp<around*|(|-s<rsub|4><around*|(|b\<odot\>z|)>|)>>>>>
  </eqnarray*>

  or\ 

  <\equation*>
    c<rsup|-1><around*|(|z|)>=b\<odot\>z+<around*|(|1-b|)>\<odot\><around*|[|x-t<around*|(|b\<odot\>z|)>|]>\<odot\>exp<around*|(|-s<around*|(|b\<odot\>z|)>|)>
  </equation*>

  The Jacobian matrix of the invertible mapping <math|c<around*|(|z|)>> is
  determined by<\footnote>
    We have useed the fact that <math|b\<odot\>x=<around*|(|x<rsub|1>,x<rsub|2>,0,0|)>>
    is constant in term of <math|x<rsub|3>> and <math|x<rsub|4>>.
  </footnote>

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>z<rsub|1>|\<partial\>x<rsub|1>>>|<cell|=>|<cell|1,<frac|\<partial\>z<rsub|1>|\<partial\>x<rsub|2>>=<frac|\<partial\>z<rsub|1>|\<partial\>x<rsub|3>>=<frac|\<partial\>z<rsub|1>|\<partial\>x<rsub|4>>=0>>|<row|<cell|<frac|\<partial\>z<rsub|2>|\<partial\>x<rsub|2>>>|<cell|=>|<cell|1,<frac|\<partial\>z<rsub|2>|\<partial\>x<rsub|1>>=<frac|\<partial\>z<rsub|2>|\<partial\>x<rsub|3>>=<frac|\<partial\>z<rsub|2>|\<partial\>x<rsub|4>>=0>>|<row|<cell|<frac|\<partial\>z<rsub|3>|\<partial\>x<rsub|3>>>|<cell|=>|<cell|exp<around*|(|s<rsub|3><around*|(|b\<odot\>x|)>|)>,<frac|\<partial\>z<rsub|3>|\<partial\>x<rsub|4>>=0>>|<row|<cell|<frac|\<partial\>z<rsub|4>|\<partial\>x<rsub|4>>>|<cell|=>|<cell|exp<around*|(|s<rsub|4><around*|(|b\<odot\>x|)>|)>,<frac|\<partial\>z<rsub|4>|\<partial\>x<rsub|3>>=0>>>>
  </eqnarray*>

  That is,

  <\equation*>
    <frac|\<partial\>z|\<partial\>x>=<around*|(|<tabular|<tformat|<table|<row|<cell|1>|<cell|0>|<cell|0>|<cell|0>>|<row|<cell|0>|<cell|1>|<cell|0>|<cell|0>>|<row|<cell|**?>|<cell|?>|<cell|exp<around*|(|s<rsub|3><around*|(|b\<odot\>x|)>|)>>|<cell|0>>|<row|<cell|?>|<cell|?>|<cell|0>|<cell|exp<around*|(|s<rsub|4><around*|(|b\<odot\>x|)>|)>>>>>>|)>
  </equation*>

  and the log of absolute value of its determinant is

  <\eqnarray*>
    <tformat|<table|<row|<cell|log<around*|[|<around*|\||det<around*|(|<frac|\<partial\>z|\<partial\>x>|)>|\|>|]>>|<cell|=>|<cell|log<around*|\||1\<cdot\>1\<cdot\>exp<around*|(|s<rsub|3><around*|(|b\<odot\>x|)>|)>\<cdot\>exp<around*|(|s<rsub|4><around*|(|b\<odot\>x|)>|)>|\|>>>|<row|<cell|>|<cell|=>|<cell|0+0+s<rsub|3><around*|(|b\<odot\>x|)>+s<rsub|4><around*|(|b\<odot\>x|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|4><around*|[|<around*|(|1-b|)>\<odot\>s<around*|(|b\<odot\>x|)>|]><rsub|i>>>>>
  </eqnarray*>

  where <math|<around*|[|<around*|(|1-b|)>\<odot\>s<around*|(|b\<odot\>x|)>|]><rsub|i>>
  denotes the <math|i>-th component of <math|<around*|(|1-b|)>\<odot\>s<around*|(|b\<odot\>x|)>>.

  We also have

  <\eqnarray*>
    <tformat|<table|<row|<cell|1>|<cell|=>|<cell|<big|int>p<rsub|Z><around*|(|z|)>d
    z>>|<row|<cell|>|<cell|=>|<cell|<big|int>p<rsub|Z><around*|(|c<around*|(|x|)>|)><around*|\||det<around*|(|<frac|\<partial\>z|\<partial\>x>|)>|\|>
    d x>>|<row|<cell|>|<cell|=>|<cell|<big|int>p<rsub|X><around*|(|x|)>d
    x>>|<row|<cell|>|<cell|>|<cell|\<Rightarrow\>p<rsub|X><around*|(|x|)>=p<rsub|Z><around*|(|c<around*|(|x|)>|)><around*|\||det<around*|(|<frac|\<partial\>z|\<partial\>x>|)>|\|>>>>>
  </eqnarray*>

  and thus,

  <\eqnarray*>
    <tformat|<table|<row|<cell|log p<rsub|X><around*|(|x|)>>|<cell|=>|<cell|log
    p<rsub|Z><around*|(|c<around*|(|x|)>|)>+log<around*|\||det<around*|(|<frac|\<partial\>z|\<partial\>x>|)>|\|>>>|<row|<cell|>|<cell|=>|<cell|log
    p<rsub|Z><around*|(|c<around*|(|x|)>|)>+<big|sum><rsub|i=1><rsup|4><around*|[|<around*|(|1-b|)>\<odot\>s<around*|(|b\<odot\>x|)>|]><rsub|i>>>>>
  </eqnarray*>

  \;

  \;

  \;

  \;

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|footnote-1|<tuple|1|1>>
    <associate|footnr-1|<tuple|1|1>>
  </collection>
</references>