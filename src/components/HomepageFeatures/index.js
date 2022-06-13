import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'La industria agrícola se está beneficiando de los científicos de datos',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        La ciencia de los datos está cambiando la forma en que los agricultores y los profesionales de la agricultura han estado tomando decisiones 
        <a href='https://towardsdatascience.com/6-ways-the-agricultural-industry-is-benefiting-from-data-scientists-b778d83f61db' target="_blank"> (Matthews, 2019)</a>.
      </>
    ),
  },
  {
    title: 'Qué es el Internet de las cosas?',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        La tecnología moderna ha permitido recopilar datos del suelo, el agua y los minerales de las explotaciones agrícolas, y almacenarlos en un sistema centralizado, conocido popularmente como Internet de las Cosas (IoT). IoT se refiere a la idea de conectar a Internet dispositivos interrelacionados para que puedan compartir e intercambiar datos de forma independiente 
        <a href='https://www.ibm.com/blogs/internet-of-things/what-is-the-iot/' target="_blank"> (Clark, 2016)</a>. 
      </>
    ),
  },
  {
    title: 'Agricultura digital e impulsada por los datos',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        La analítica de datos puede utilizarse en el volumen acumulado para obtener información que los agricultores pueden utilizar para optimizar su agricultura. Así, los agricultores pueden tomar decisiones agrícolas inteligentes utilizando esa información a lo largo del ciclo de producción; desde la planificación, la plantación, la cosecha, hasta su comercialización 
        <a href='https://cgspace.cgiar.org/bitstream/handle/10568/92477/GFAR-GODAN-CTA-white-paper-final.pdf' target="_blank"> (Maru, 2018).</a>
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
