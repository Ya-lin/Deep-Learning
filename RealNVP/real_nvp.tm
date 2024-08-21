<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Real NVP>>

  Let <math|z=<around*|(|z<rsub|1>,z<rsub|2>,z<rsub|3>,z<rsub|4>|)>\<sim\><with|font|cal|N><around*|(|0,1|)>>
  and <math|x=<around*|(|x<rsub|1>,x<rsub|2>,x<rsub|3>,x<rsub|4>|)>\<sim\>P<rsub|x>>.
  The goal is to design an invertible network such that it can transform
  normal distribution <math|<with|font|cal|N><around*|(|0,1|)>> to be data
  distribution <math|P<rsub|x>>.

  Give the boolean vector <math|b=<around*|(|1,1,0,0|)>> and two mappings

  <\eqnarray*>
    <tformat|<table|<row|<cell|s>|<cell|:>|<cell|\<bbb-R\><rsup|4>\<rightarrow\>\<bbb-R\><rsup|4>>>|<row|<cell|t>|<cell|:>|<cell|\<bbb-R\><rsup|4>\<rightarrow\>\<bbb-R\><rsup|4>,>>>>
  </eqnarray*>

  \ the coupling layer <math|x=c<around*|(|z|)>> is defined as the
  transformation

  <\eqnarray*>
    <tformat|<table|<row|<cell|c>|<cell|:>|<cell|\<bbb-R\><rsup|4>\<rightarrow\>\<bbb-R\><rsup|4>>>|<row|<cell|>|<cell|>|<cell|z\<rightarrow\>x=b\<odot\>z+<around*|(|1-b|)>\<odot\><around*|[|z\<odot\>exp<around*|(|s<around*|(|b\<odot\>z|)>|)>+t<around*|(|b\<odot\>z|)>|]>>>>>
  </eqnarray*>

  where <math|exp> is component-wise, and <math|\<odot\>> is the Hadamard
  component-wise product. Particularly,

  <\eqnarray*>
    <tformat|<table|<row|<cell|x<rsub|1>>|<cell|=>|<cell|z<rsub|1>>>|<row|<cell|x<rsub|2>>|<cell|=>|<cell|z<rsub|2>>>|<row|<cell|x<rsub|3>>|<cell|=>|<cell|z<rsub|3>\<cdot\>exp<around*|(|<around*|[|s<around*|(|b\<odot\>z|)>|]><rsub|3>|)>+<around*|[|t<around*|(|b\<odot\>z|)>|]><rsub|3>>>|<row|<cell|x<rsub|4>>|<cell|=>|<cell|z<rsub|4>\<cdot\>exp<around*|(|<around*|[|s<around*|(|b\<odot\>z|)>|]><rsub|4>|)>+<around*|[|t<around*|(|b\<odot\>z|)>|]><rsub|4>>>>>
  </eqnarray*>

  where <math|><math|<around*|[|s<around*|(|b\<odot\>z|)>|]><rsub|i>> and
  <math|<around*|[|t<around*|(|b\<odot\>z|)>|]><rsub|i>> is the <math|i>-th
  component of <math|s<around*|(|b\<odot\>z|)>> and
  <math|t<around*|(|b\<odot\>z|)>>, respectively. By arranging terms,

  <\eqnarray*>
    <tformat|<table|<row|<cell|z<rsub|1>>|<cell|=>|<cell|x<rsub|1>>>|<row|<cell|z<rsub|2>>|<cell|=>|<cell|x<rsub|2>>>|<row|<cell|z<rsub|3>>|<cell|=>|<cell|<around*|(|x<rsub|3>-<around*|[|t<around*|(|b\<odot\>z|)>|]><rsub|3>|)>\<cdot\>exp<around*|(|-<around*|[|s<around*|(|b\<odot\>z|)>|]><rsub|3>|)>>>|<row|<cell|z<rsub|4>>|<cell|=>|<cell|<around*|(|x<rsub|4>-<around*|[|t<around*|(|b\<odot\>z|)>|]><rsub|4>|)>\<cdot\>exp<around*|(|-<around*|[|s<around*|(|b\<odot\>z|)>|]><rsub|4>|)>>>>>
  </eqnarray*>

  Note that <math|b\<odot\>z=<around*|(|1,1,0,0|)>\<odot\><around*|(|z<rsub|1>,z<rsub|2>,z<rsub|3>,z<rsub|4>|)>=<around*|(|z<rsub|1>,z<rsub|2>,0,0|)>=<around*|(|x<rsub|1>,x<rsub|2>,0,0|)>=b\<odot\>x>,
  we have the inverse mapping <math|z=c<rsup|-1><around*|(|x|)>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|z<rsub|1>>|<cell|=>|<cell|x<rsub|1>>>|<row|<cell|z<rsub|2>>|<cell|=>|<cell|x<rsub|2>>>|<row|<cell|z<rsub|3>>|<cell|=>|<cell|<around*|(|x<rsub|3>-<around*|[|t<around*|(|b\<odot\>x|)>|]><rsub|3>|)>\<cdot\>exp<around*|(|-<around*|[|s<around*|(|b\<odot\>x|)>|]><rsub|3>|)>>>|<row|<cell|z<rsub|4>>|<cell|=>|<cell|<around*|(|x<rsub|4>-<around*|[|t<around*|(|b\<odot\>x|)>|]><rsub|4>|)>\<cdot\>exp<around*|(|-<around*|[|s<around*|(|b\<odot\>x|)>|]><rsub|4>|)>>>>>
  </eqnarray*>

  or\ 

  <\equation*>
    c<rsup|-1><around*|(|x|)>=b\<odot\>x+<around*|(|1-b|)><around*|(|x-t<around*|(|b\<odot\>x|)>|)>\<odot\>exp<around*|(|s<around*|(|b\<odot\>x|)>|)>
  </equation*>

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