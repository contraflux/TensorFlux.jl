// import Layout from '../components/Layout/Layout';
// import CodeBlock from '../components/CodeBlock/CodeBlock';
// import Math from '../components/Math/Math';
import { Link } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import style from './home.module.css';
import github from '../assets/github.svg';

export default function GettingStarted() {
    return (
        <Layout sidebar={false}>
            <div className={style.section}>
                <div className={style.heading}>
                    <p className={style.title}>Differential Geometry Engine for Julia</p>
                    <p className={style.subtitle}>
                        TensorFlux.jl provides tools for tensor algebra, tensor calculus,
                        differential geometry, and more
                    </p>
                    <div className={style.menu}>
                        <Link className={`${style.item} ${style.highlight}`} to='/learn/getting-started'>
                            <p>Get Started</p>
                            <p className={style.icon}>&#8594;</p>
                        </Link>
                        <a className={style.item} href='https://github.com/contraflux/TensorFlux.jl' target='_blank'>
                            <p>View on GitHub</p>
                            <img className={style.icon} src={github}></img>
                        </a>
                    </div>
                </div>
            </div>
            <div className={`${style.section} ${style.dark}`}>
                <div className={style.subheading}>
                    <p className={`${style.title} ${style.center}`}>Do this cool action!</p>
                    <p className={`${style.subtitle} ${style.center}`}>
                        Text explaining how this action works and why its useful!
                    </p>
                </div>
            </div>
            <div className={style.section}>
                <div className={style.subheading}>
                    <p className={`${style.title} ${style.center}`}>Do this cool action!</p>
                    <p className={`${style.subtitle} ${style.center}`}>
                        Text explaining how this action works and why its useful!
                    </p>
                </div>
            </div>
        </Layout>
    );
}